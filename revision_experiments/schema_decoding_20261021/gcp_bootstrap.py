"""GCP H100 setup, model downloads, live format checks, and optional inference.

The controller uses the standard library. Dependency-heavy worker steps execute
in a new dedicated environment, never in the workstation's native environment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
VERSION = "gcp-allinone-v3-cu129-compact-20261021"
UV_VERSION = "0.13.4"
VLLM_VERSION = "0.21.0+cu129"
VLLM_WHEEL = "https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0%2Bcu129-cp38-abi3-manylinux_2_34_x86_64.whl#sha256=920777691e340df7a8328adfb1e57b9996dbb537edfb654dd32f70844f5f423d"
PYTHON_VERSION = "3.12.15"
MODELS = {
    "qwen": {"repo": "Qwen/Qwen3-VL-8B-Instruct", "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "weight_bytes": 17534339512},
    "gemma": {"repo": "google/gemma-3-12b-it", "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80", "weight_bytes": 24374793024},
    "internvl": {"repo": "OpenGVLab/InternVL3-8B", "revision": "853e3a797a661694b1b8ece0cb72dc2b23e3dac9", "weight_bytes": 15888831920},
}
SYSTEM_PACKAGES = ("python3-venv", "ca-certificates", "git", "build-essential", "libnuma1", "libgl1", "libglib2.0-0")
DRIVER_URL = "https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz"
SMOKE_CASES = 2


class RebootRequired(RuntimeError):
    pass


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def selected_models(names):
    if names == ["all"]:
        return list(MODELS)
    if not names or len(set(names)) != len(names) or any(name not in MODELS for name in names):
        raise ValueError("Use --models all or unique names from qwen gemma internvl")
    return names


def default_bundle():
    return HERE if (HERE / "bundle_manifest.json").exists() else HERE / "cloud_bundle"


def load_client(bundle):
    spec = importlib.util.spec_from_file_location("frozen_schema_client", Path(bundle) / "schema_decoding.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_gcp_host():
    if platform.system() != "Linux" or platform.machine() != "x86_64" or "microsoft" in platform.release().lower():
        raise RuntimeError("Installation is restricted to a GCP Linux x86_64 VM, not the existing WSL workstation")
    request = urllib.request.Request("http://169.254.169.254/computeMetadata/v1/instance/id", headers={"Metadata-Flavor": "Google"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=3) as response:
            instance_id = response.read().decode().strip()
    except OSError as exc:
        raise RuntimeError("GCP metadata is unavailable; refusing OS/package installation on an unverified host") from exc
    if not instance_id.isdigit():
        raise RuntimeError("Invalid GCP instance metadata")
    os_release = platform.freedesktop_os_release()
    if os_release.get("ID") != "ubuntu" or os_release.get("VERSION_ID") not in ("22.04", "24.04"):
        raise RuntimeError("Use Ubuntu 22.04 or 24.04 for this pinned installer")
    return {"instance_id": instance_id, "os": os_release, "kernel": platform.release()}


def runtime_env(root, venv=None):
    root = Path(root)
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("TRANSFORMERS_CACHE", None)
    for name, relative in {
        "HF_HOME": "cache/huggingface", "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub",
        "HF_HUB_CACHE": "cache/huggingface/hub", "HF_MODULES_CACHE": "cache/huggingface/modules",
        "VLLM_CACHE_ROOT": "cache/vllm", "UV_CACHE_DIR": "cache/uv", "UV_PYTHON_INSTALL_DIR": "python",
        "TMPDIR": "tmp", "PIP_CACHE_DIR": "cache/pip",
    }.items():
        env[name] = str(root / relative)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["PATH"] = env["CUDA_HOME"] + "/bin" + os.pathsep + env.get("PATH", "")
    if venv:
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = str(Path(venv) / "bin") + os.pathsep + env.get("PATH", "")
    return env


def serving_argv(venv, model, port):
    info = MODELS[model]
    return [str(Path(venv) / "bin/vllm"), "serve", info["repo"],
            "--revision", info["revision"], "--tokenizer-revision", info["revision"], "--code-revision", info["revision"],
            "--served-model-name", info["repo"], "--host", "127.0.0.1", "--port", str(port),
            "--dtype", "bfloat16", "--gpu-memory-utilization", "0.90", "--max-model-len", "16384",
            "--max-num-seqs", "1", "--limit-mm-per-prompt", '{"image":2}',
            "--structured-outputs-config", '{"backend":"xgrammar","disable_any_whitespace":true}',
            "--generation-config", "vllm", "--model-impl", "vllm", "--load-format", "safetensors",
            "--seed", "0", "--enforce-eager", "--trust-remote-code"]


def ensure_free_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # Closed HTTP connections can leave TIME_WAIT without a listening server.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is already in use; no existing server will be killed") from exc


def ensure_idle_gpu(env):
    completed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, env=env, check=True)
    pids = [line.strip() for line in completed.stdout.splitlines() if line.strip().isdigit()]
    if pids:
        raise RuntimeError(f"GPU already has compute processes {pids}; refusing to interrupt them")


def wait_idle_gpu(env, timeout=60):
    started = time.monotonic()
    while True:
        try:
            ensure_idle_gpu(env)
            return
        except RuntimeError:
            if time.monotonic() - started >= timeout:
                raise
            time.sleep(2)


def stop_owned_server(process, timeout=90):
    # A crashed parent can leave workers in its process group; signal only our own group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)


def required_free_gib(root, models, initial_minimum):
    root = Path(root)
    missing = []
    for model in models:
        marker = root / "models" / f"{model}.json"
        try:
            snapshot = Path(read_json(marker)["snapshot_path"])
            cached = sum(path.stat().st_size for path in snapshot.rglob("*.safetensors")) if snapshot.is_dir() else 0
        except (OSError, ValueError, KeyError):
            cached = 0
        if cached < MODELS[model]["weight_bytes"]:
            missing.append(model)
    environment_missing = not (root / "environment.freeze.txt").exists()
    needed = 10 + sum((MODELS[model]["weight_bytes"] + 1024**3 - 1) // 1024**3 for model in missing)
    if environment_missing:
        needed += 40
    # Do not demand another 120 GiB after the models and environment are cached.
    return max(needed, initial_minimum) if environment_missing and len(missing) == len(models) else needed


def verify_smoke_rows(rows, schema, validator):
    counts = {condition: 0 for condition in ("json_object", "json_schema")}
    schema_truncated = 0
    for _, row in rows:
        if row.get("transport_ok") is not True:
            raise RuntimeError("Live smoke test has a transport error")
        counts[row["condition"]] += 1
        if row["condition"] == "json_schema":
            if row.get("finish_reason") == "length":
                # A valid grammar prefix can be truncated; it is a measured model outcome.
                schema_truncated += 1
                continue
            try:
                # Validate raw model text, not extracted/normalized/repaired content.
                validator(json.loads(row["origin_output"]), schema)
            except (ValueError, TypeError) as exc:
                raise RuntimeError("JSON Schema output is not raw JSON; do not run the full experiment") from exc
    if counts != {"json_object": SMOKE_CASES, "json_schema": SMOKE_CASES}:
        raise RuntimeError(f"Live smoke test needs exactly two cases per condition, got {counts}")
    return {"passed": True, "rows": sum(counts.values()), "by_condition": counts,
            "semantic_plan_correctness_required": False, "schema_truncated": schema_truncated,
            "raw_schema_enforcement_verified": schema_truncated == 0}


def export_artifacts(root, run_id):
    root = Path(root)
    destination = root / "exports" / f"inference_artifacts_{run_id}.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    allowed = ("outputs", "smoke", "serving", "checks", "host.json", "bootstrap_project.json", "environment.freeze.txt",
               "READY.json", "INFERENCE_COMPLETE.json", "status.json")
    included = []
    with destination.open("xb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz") as tar:
            for name in allowed:
                path = root / name
                if not path.exists():
                    continue
                candidates = [path] + (list(path.rglob("*")) if path.is_dir() else [])
                for candidate in candidates:
                    if candidate.is_symlink():
                        raise RuntimeError("Artifact export refuses symbolic links")
                    tar.add(candidate, arcname=str(candidate.relative_to(root)), recursive=False)
                    if candidate.is_file():
                        included.append(str(candidate.relative_to(root)))
    report = {"archive": str(destination), "sha256": sha256(destination), "files": included,
              "excludes": ["cache/Hugging Face credentials", "model weights", "virtual environments", "temporary files", "server/install logs"]}
    write_json(destination.with_suffix(".manifest.json"), report)
    return report


def worker(args):
    """Dependency-heavy operations run only with the newly created venv Python."""
    import importlib.metadata as metadata

    root = args.work_root
    if args.worker == "gpu":
        import torch
        import vllm
        # The Python package can import even when its compiled CUDA extension cannot.
        import vllm._C
        from vllm.platforms import current_platform

        if metadata.version("vllm") != VLLM_VERSION or not torch.cuda.is_available():
            raise RuntimeError("Pinned vLLM or CUDA availability check failed")
        if not current_platform.is_cuda():
            raise RuntimeError("The compiled vLLM CUDA platform is unavailable")
        if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This installer requires one visible H100 GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported by the current CUDA stack")
        free, total = torch.cuda.mem_get_info()
        if free < 60 * 1024**3:
            raise RuntimeError("At least 60 GiB of free H100 memory is required")
        matrix = torch.ones((128, 128), dtype=torch.bfloat16, device="cuda")
        value = float((matrix @ matrix)[0, 0].item())
        if value != 128.0:
            raise RuntimeError("BF16 GPU computation returned an unexpected result")
        names = ("vllm", "torch", "torchvision", "transformers", "huggingface-hub", "tokenizers", "timm", "jsonschema", "xgrammar")
        result = {"gpu": torch.cuda.get_device_name(0), "free_bytes": free, "total_bytes": total, "python": sys.version,
                  "torch_cuda": torch.version.cuda, "bf16_matmul": value,
                  "packages": {name: metadata.version(name) for name in names}}
    elif args.worker == "authorize":
        from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_url, login
        from huggingface_hub.errors import HfHubHTTPError

        result = {}
        for name in selected_models(args.models):
            info = MODELS[name]
            for attempt in range(2):
                try:
                    model = HfApi().model_info(info["repo"], revision=info["revision"])
                    weight = next(item.rfilename for item in model.siblings if item.rfilename.endswith(".safetensors"))
                    get_hf_file_metadata(hf_hub_url(info["repo"], weight, revision=info["revision"]), token=get_token())
                    if model.sha != info["revision"]:
                        raise RuntimeError("Unexpected model snapshot revision")
                    result[name] = {**info, "weight_access_verified": True}
                    break
                except HfHubHTTPError as exc:
                    status = getattr(exc.response, "status_code", None)
                    if status not in (401, 403):
                        raise
                    if attempt or os.environ.get("HF_TOKEN") or not sys.stdin.isatty():
                        raise RuntimeError(f"Access denied for {info['repo']}; accept its license/access request and use hf auth login or HF_TOKEN") from exc
                    print(f"Access is required for {info['repo']}. Token entry is hidden and is not logged.", flush=True)
                    token = getpass.getpass("Hugging Face read token: ")
                    login(token=token, add_to_git_credential=False)
                    del token
    elif args.worker == "download":
        from huggingface_hub import snapshot_download

        info = MODELS[args.model]
        snapshot = Path(snapshot_download(info["repo"], revision=info["revision"],
                        allow_patterns=["*.safetensors", "*.json", "*.model", "*.py", "*.txt", "*.jinja", "*.tiktoken", "tokenizer*"],
                        max_workers=4))
        weights = list(snapshot.rglob("*.safetensors"))
        if not weights or sum(path.stat().st_size for path in weights) < 5 * 1024**3:
            raise RuntimeError("Snapshot is incomplete or does not contain expected model weights")
        config = read_json(snapshot / "config.json")
        if config.get("quantization_config"):
            raise RuntimeError("Quantized model configuration is not allowed in the BF16 comparison")
        result = {**info, "snapshot_path": str(snapshot), "weight_files": len(weights),
                  "weight_bytes": sum(path.stat().st_size for path in weights)}
    elif args.worker == "metadata":
        snapshot = Path(read_json(root / "models" / f"{args.model}.json")["snapshot_path"])
        configs = {}
        for name in ("config.json", "preprocessor_config.json", "processor_config.json", "tokenizer_config.json", "chat_template.json", "chat_template.jinja"):
            path = snapshot / name
            if path.is_file():
                configs[name] = {"sha256": sha256(path), "content": read_json(path) if path.suffix == ".json" else path.read_text(encoding="utf-8")}
        if not configs:
            raise RuntimeError("No model/processor configuration was captured")
        result = {"model_revision": MODELS[args.model]["revision"], "vllm_version": metadata.version("vllm"),
                  "dtype": "bfloat16", "server_command": shlex.join(serving_argv(root / "venv-vllm", args.model, args.port)),
                  "processor_revision": MODELS[args.model]["revision"],
                  "chat_template": {"setting": "model default; no custom override", "configs": configs},
                  "image_preprocessing": {"setting": "model/vLLM default; no processor overrides", "source_configs": configs,
                                          "client_images": "unchanged input.png then input_mask.png"},
                  "request_seed": 0, "server_seed": 0, "temperature": 0.0, "max_tokens": 1024,
                  "environment_lock_sha256": sha256(root / "environment.freeze.txt")}
        gpu = read_json(root / "checks/gpu.json")
        result.update(python=gpu["python"], packages=gpu["packages"], torch_cuda=gpu["torch_cuda"],
                      driver=read_json(root / "checks/driver.json")["driver"])
    elif args.worker == "smoke":
        import jsonschema

        client = load_client(args.bundle)
        _, rows = client.load_frozen_generation(root / "smoke" / args.model, args.bundle)
        try:
            result = verify_smoke_rows(rows, read_json(args.bundle / "procedural_plan_schema.json"), jsonschema.validate)
        except jsonschema.ValidationError as exc:
            raise RuntimeError("Server accepted JSON Schema but returned a schema-invalid response; stopped without repair") from exc
        probe = read_json(root / "checks/format_probe" / args.model / "report.json")
        if probe.get("passed") is not True:
            raise RuntimeError("The independent grammar-enforcement probe is missing")
        result["independent_grammar_probe_passed"] = True
    else:
        raise ValueError(args.worker)
    write_json(args.worker_output, result)
    print(json.dumps({"worker": args.worker, "model": args.model, "status": "passed"}), flush=True)
    return 0


class Bootstrap:
    def __init__(self, args):
        self.args = args
        self.root = args.work_root.resolve()
        self.bundle = args.bundle.resolve()
        self.models = selected_models(args.models)
        self.venv = self.root / "venv-vllm"
        self.python = self.venv / "bin/python"
        self.env = runtime_env(self.root, self.venv)
        self.run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def log(self, stage, message):
        value = {"time": utc_now(), "run_id": self.run_id, "stage": stage, "message": message}
        print(json.dumps(value, ensure_ascii=False), flush=True)
        if self.root.exists():
            with (self.root / "controller.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    def execute(self, argv, stage, env=None, capture=False):
        self.log(stage, "Running " + shlex.join([str(part) for part in argv]))
        process = subprocess.Popen([str(part) for part in argv], env=env or self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = []
        with (self.root / "logs" / f"{self.run_id}_{stage}.log").open("a", encoding="utf-8") as handle:
            try:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    if capture:
                        output.append(line)
                    else:
                        print(line, end="", flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait(timeout=30)
                raise
        if code:
            raise RuntimeError(f"{stage} failed with exit code {code}; see {self.root / 'logs'}")
        return "".join(output)

    def initialize(self):
        client = load_client(self.bundle)
        client.require_separate(self.root, self.bundle)
        manifest = client.load_bundle(self.bundle)
        if len(manifest["cases"]) < SMOKE_CASES:
            raise RuntimeError("The input bundle must have at least two cases")
        marker = self.root / "bootstrap_project.json"
        config = {"version": VERSION, "requirements_sha256": sha256(HERE / "gcp_requirements.txt"),
                  "dependency_lock_sha256": sha256(HERE / "gcp_dependencies.lock.txt"),
                  "bundle_manifest_sha256": sha256(self.bundle / "bundle_manifest.json"), "uv_version": UV_VERSION,
                  "python": PYTHON_VERSION, "torch_backend": "cu129", "controller_sha256": sha256(__file__)}
        if marker.exists() and read_json(marker) != config:
            raise RuntimeError("This work directory belongs to a different setup or bundle; use a new --work-root")
        if self.root.exists() and not marker.exists() and any(self.root.iterdir()):
            raise RuntimeError("Refusing to reuse an existing nonempty, unowned work directory")
        self.root.mkdir(parents=True, exist_ok=True)
        for relative in ("logs", "tmp", "cache", "models", "checks", "serving", "smoke", "outputs"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        write_json(marker, config)
        return client, manifest

    def setup_system(self):
        if (self.root / "system_ready.json").exists():
            self.log("system", "Previously completed OS prerequisites retained")
            return
        sudo = [] if os.geteuid() == 0 else ["sudo", "-n"]
        env = dict(self.env, DEBIAN_FRONTEND="noninteractive")
        prefix = sudo + ["env", "DEBIAN_FRONTEND=noninteractive"]
        self.execute(prefix + ["apt-get", "update"], "apt_update", env)
        self.execute(prefix + ["apt-get", "install", "-y", "--no-install-recommends", *SYSTEM_PACKAGES], "apt_install", env)
        write_json(self.root / "system_ready.json", {"packages": SYSTEM_PACKAGES, "time": utc_now()})

    def setup_driver(self):
        marker = self.root / "NEEDS_REBOOT.json"
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if marker.exists() and read_json(marker)["boot_id"] == boot_id:
            raise RebootRequired("Driver installation requires a reboot. Reboot the GCP VM, reconnect, and rerun the same command")
        probe = subprocess.run(["nvidia-smi"], capture_output=True, text=True) if shutil.which("nvidia-smi") else None
        if probe is not None and probe.returncode == 0:
            ensure_idle_gpu(self.env)
            driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"], text=True)
            name = driver.split(",", 1)[-1]
            if "H100" not in name:
                raise RuntimeError("The VM is not the requested H100 instance")
            version = tuple(int(part) for part in driver.split(",", 1)[0].strip().split("."))
            if version >= (580, 95, 5):
                write_json(self.root / "checks/driver.json", {"nvidia_smi": probe.stdout, "driver": driver.strip()})
                self.log("driver", "Existing H100 driver passed; no driver change")
                return
        if not self.args.install_driver:
            raise RuntimeError("NVIDIA driver unavailable/outdated. Rerun with --install-driver on this GCP VM")
        for path in Path("/sys/firmware/efi/efivars").glob("SecureBoot-*"):
            if path.read_bytes()[-1:] == b"\x01":
                raise RuntimeError("Secure Boot requires a signed-driver installation; the automated installer will not change Secure Boot")
        installer = self.root / "cuda_installer.pyz"
        with urllib.request.urlopen(DRIVER_URL, timeout=120) as response:
            installer.write_bytes(response.read())
        write_json(self.root / "checks/driver_installer.json", {"url": DRIVER_URL, "sha256": sha256(installer)})
        write_json(marker, {"boot_id": boot_id, "time": utc_now(), "reason": "NVIDIA driver installation"})
        sudo = [] if os.geteuid() == 0 else ["sudo", "-n"]
        try:
            self.execute(sudo + ["python3", str(installer), "install_driver", "--installation-mode=repo", "--installation-branch=prod"], "driver_install")
        except BaseException:
            marker.unlink(missing_ok=True)
            raise
        raise RebootRequired("Driver installed. Reboot the VM and rerun the same command; existing model outputs will not be removed")

    def setup_environment(self):
        self.setup_cuda_toolkit()
        lock = self.root / "environment.freeze.txt"
        if lock.exists():
            if not self.python.is_file():
                raise RuntimeError("The locked venv is missing; use a new work directory")
            self.execute([self.python, "-m", "pip", "check"], "pip_check")
            current = self.execute([self.python, "-m", "pip", "freeze"], "pip_freeze_check", capture=True)
            if set(current.splitlines()) != set(lock.read_text().splitlines()):
                raise RuntimeError("Installed environment differs from the frozen dependency list")
            self.log("venv", "Existing dedicated environment verified; packages not upgraded")
            return
        bootstrap_python = self.root / "bootstrap-venv/bin/python"
        if not bootstrap_python.exists():
            self.execute([sys.executable, "-m", "venv", self.root / "bootstrap-venv"], "create_bootstrap_venv")
        base_env = runtime_env(self.root, self.root / "bootstrap-venv")
        self.execute([bootstrap_python, "-m", "pip", "install", "--no-cache-dir", f"uv=={UV_VERSION}"], "install_uv", base_env)
        uv = self.root / "bootstrap-venv/bin/uv"
        if not self.python.exists():
            self.execute([uv, "venv", "--python", PYTHON_VERSION, "--seed", "--managed-python", self.venv], "create_vllm_venv", base_env)
        self.execute([uv, "pip", "install", "--python", self.python, "--torch-backend=cu129", "-r", HERE / "gcp_dependencies.lock.txt"], "install_dependencies")
        self.execute([self.python, "-m", "pip", "check"], "pip_check")
        frozen = self.execute([self.python, "-m", "pip", "freeze"], "pip_freeze", capture=True)
        lock.write_text(frozen, encoding="utf-8")

    def setup_cuda_toolkit(self):
        # FlashInfer builds its sampling kernels on first use, including warmup.
        nvcc = Path("/usr/local/cuda-12.9/bin/nvcc")
        if not nvcc.exists() or not Path("/usr/local/cuda-12.9/include/curand.h").exists():
            distro = "ubuntu" + platform.freedesktop_os_release()["VERSION_ID"].replace(".", "")
            url = f"https://developer.download.nvidia.com/compute/cuda/repos/{distro}/x86_64/cuda-keyring_1.1-1_all.deb"
            keyring = self.root / "cuda-keyring_1.1-1_all.deb"
            with urllib.request.urlopen(url, timeout=120) as response:
                keyring.write_bytes(response.read())
            sudo = [] if os.geteuid() == 0 else ["sudo", "-n"]
            self.execute(sudo + ["dpkg", "-i", keyring], "cuda_keyring")
            self.execute(sudo + ["apt-get", "update"], "cuda_apt_update")
            packages = ["cuda-nvcc-12-9", "cuda-cudart-dev-12-9", "cuda-cccl-12-9", "libcurand-dev-12-9"]
            prefix = sudo + ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get"]
            plan = self.execute(prefix + ["install", "--simulate", "--no-install-recommends", *packages], "cuda_plan", capture=True)
            if any(line.startswith(("Inst nvidia-driver", "Inst cuda-drivers", "Remv ")) for line in plan.splitlines()):
                raise RuntimeError("CUDA compiler installation would change the driver or remove packages")
            self.execute(prefix + ["install", "-y", "--no-install-recommends", *packages], "cuda_install")
        version = self.execute([nvcc, "--version"], "nvcc_version", capture=True)
        if "release 12.9" not in version:
            raise RuntimeError("CUDA compiler must match the pinned CUDA 12.9 runtime")
        write_json(self.root / "checks/cuda_toolkit.json", {"nvcc": str(nvcc), "version": version})

    def call_worker(self, stage, model=None, output=None):
        output = output or self.root / "checks" / f"{stage}.json"
        argv = [self.python, "-u", __file__, "--worker", stage, "--worker-output", output,
                "--work-root", self.root, "--bundle", self.bundle, "--models", *self.models, "--port", str(self.args.port)]
        if model:
            argv += ["--model", model]
        self.execute(argv, stage + ("_" + model if model else ""))
        return read_json(output)

    def stable_metadata(self, model):
        candidate = self.root / "checks" / f"serving_candidate_{model}.json"
        result = self.call_worker("metadata", model, candidate)
        final = self.root / "serving" / f"{model}.json"
        if final.exists() and read_json(final) != result:
            raise RuntimeError("Serving metadata changed; refusing to resume old generations under new settings")
        write_json(final, result)
        return final

    @contextmanager
    def server(self, model):
        ensure_free_port(self.args.port)
        ensure_idle_gpu(self.env)
        command = serving_argv(self.venv, model, self.args.port)
        log_path = self.root / "logs" / f"{self.run_id}_vllm_{model}.log"
        self.log("server", f"Starting {model}; log={log_path}")
        with log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(command, env=self.env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(self.root / "checks" / f"server_{model}.json", {"pid": process.pid, "command": command, "log": str(log_path)})
            try:
                started = time.monotonic()
                last_notice = started
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"vLLM exited with {process.returncode}; see {log_path}")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{self.args.port}/v1/models", timeout=3) as response:
                            inventory = json.loads(response.read())
                        if MODELS[model]["repo"] in {item["id"] for item in inventory.get("data", [])}:
                            break
                    except (OSError, ValueError):
                        pass
                    if time.monotonic() - started > self.args.startup_timeout_sec:
                        raise RuntimeError(f"vLLM startup timeout; see {log_path}")
                    if time.monotonic() - last_notice > 30:
                        self.log("server", f"Waiting for {model}, {int(time.monotonic() - started)} seconds")
                        last_notice = time.monotonic()
                    time.sleep(2)
                yield
            finally:
                stop_owned_server(process)
                wait_idle_gpu(self.env)
                self.log("server", f"Stopped only the {model} server started by this controller")

    def inference(self, model, smoke):
        output = self.root / ("smoke" if smoke else "outputs") / model
        argv = [self.python, "-u", self.bundle / "schema_decoding.py", "infer", "--bundle", self.bundle,
                "--base-url", f"http://127.0.0.1:{self.args.port}/v1", "--model", MODELS[model]["repo"],
                "--serving-config", self.root / "serving" / f"{model}.json", "--condition", "both",
                "--temperature", "0", "--max-tokens", "1024", "--seed", "0", "--output-root", output]
        if smoke:
            argv += ["--max-cases", str(SMOKE_CASES)]
        if (output / "run_config.json").exists():
            argv.append("--resume")
        if self.args.retry_api_errors:
            argv.append("--retry-api-errors")
        self.execute(argv, ("smoke_infer_" if smoke else "full_infer_") + model)

    def capability_probe(self, model):
        client = load_client(self.bundle)
        entry = client.load_bundle(self.bundle)["cases"][0]
        root = self.root / "checks/format_probe" / model
        config = {"serving_sha256": sha256(self.root / "serving" / f"{model}.json"),
                  "probe_version": "fixed-status-adversarial-v1", "case_id": entry["case_id"]}
        report = root / "report.json"
        if report.exists():
            old = read_json(report)
            if old.get("config") != config or not old.get("passed"):
                raise RuntimeError("A different or failed format probe requires review")
            return
        root.mkdir(parents=True, exist_ok=False)
        images = [part for part in client.messages_for_entry(self.bundle, entry)[0]["content"] if part["type"] == "image_url"]
        messages = [{"role": "user", "content": [{"type": "text", "text": "Return a JSON object whose status is UNCONSTRAINED. Do not use ready."}, *images]}]
        schema = {"type": "object", "properties": {"status": {"type": "string", "enum": ["ready"]}},
                  "required": ["status"], "additionalProperties": False}
        payload = client.build_payload(messages, MODELS[model]["repo"], "json_schema", schema, max_tokens=32)
        write_json(root / "request.json", client.request_record(payload, entry))
        row = client.call_api(f"http://127.0.0.1:{self.args.port}/v1", payload, 180, root)
        write_json(root / "row.json", row)
        try:
            passed = row["transport_ok"] and json.loads(row["origin_output"]) == {"status": "ready"}
        except (ValueError, TypeError):
            passed = False
        write_json(report, {"passed": passed, "config": config, "test_only": True,
                           "semantic_task_correctness_assessed": False})
        if not passed:
            raise RuntimeError("Server did not enforce the adversarial fixed-status JSON Schema")

    def run(self):
        host = verify_gcp_host()
        client, manifest = self.initialize()
        with client.run_lock(self.root):
            # Archive stale readiness markers so a failed new check cannot look ready.
            if (self.root / "READY.json").exists():
                (self.root / "READY.json").rename(self.root / "checks" / f"READY_previous_{self.run_id}.json")
            write_json(self.root / "host.json", host)
            write_json(self.root / "status.json", {"status": "checking", "run_id": self.run_id})
            try:
                needed = required_free_gib(self.root, self.models, self.args.min_free_gib)
                self.log("disk", f"Required free disk for remaining work: {needed} GiB")
                if shutil.disk_usage(self.root).free < needed * 1024**3:
                    raise RuntimeError(f"Need at least {needed} GiB free disk for remaining work; files will not be deleted automatically")
                self.setup_system()
                self.setup_driver()
                self.setup_environment()
                self.call_worker("gpu")
                # Verify gated weight access before downloading any model weights.
                self.call_worker("authorize")
                for index, model in enumerate(self.models, 1):
                    self.log("download", f"{index}/{len(self.models)} {model}")
                    self.call_worker("download", model, self.root / "models" / f"{model}.json")
                    self.stable_metadata(model)
                reports = {}
                for index, model in enumerate(self.models, 1):
                    self.log("live_check", f"{index}/{len(self.models)} {model}: two images, two cases, both output formats")
                    with self.server(model):
                        self.capability_probe(model)
                        self.inference(model, smoke=True)
                    reports[model] = self.call_worker("smoke", model, self.root / "checks" / f"smoke_{model}.json")
                ready = {"status": "ready", "run_id": self.run_id, "models": self.models, "case_count": manifest["case_count"],
                         "full_inference_calls": manifest["case_count"] * 2 * len(self.models), "smoke": reports,
                         "native_execution_on_gcp": False, "live_h100_checks_passed": True}
                write_json(self.root / "READY.json", ready)
                if self.args.action == "run":
                    for model in self.models:
                        with self.server(model):
                            self.inference(model, smoke=False)
                    ready["status"] = "inference_completed"
                    write_json(self.root / "INFERENCE_COMPLETE.json", ready)
                write_json(self.root / "status.json", ready)
                exported = export_artifacts(self.root, self.run_id)
                self.log("export", f"Transfer package (no tokens/model caches): {exported['archive']}")
                self.log("complete", f"{ready['status']}; retain the input bundle and {self.root / 'outputs'} for workstation evaluation")
                return 0
            except BaseException as exc:
                write_json(self.root / "status.json", {"status": "reboot_required" if isinstance(exc, RebootRequired) else "failed",
                           "run_id": self.run_id, "error_type": type(exc).__name__, "detail": str(exc)})
                raise


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bundle", type=Path, default=default_bundle())
    value.add_argument("--work-root", type=Path, default=Path.home() / "vlabench_revision_20261021")
    value.add_argument("--models", nargs="+", default=["all"])
    value.add_argument("--action", choices=("ready", "run"), default="ready")
    value.add_argument("--install-driver", action="store_true", help="Permit Google's driver installer if a working H100 driver is missing; reboot may be required")
    value.add_argument("--port", type=int, default=8000)
    value.add_argument("--startup-timeout-sec", type=int, default=1200)
    value.add_argument("--min-free-gib", type=int, default=120)
    value.add_argument("--retry-api-errors", action="store_true")
    value.add_argument("--plan", action="store_true", help="Show the plan without files, network, installation, downloads, or GPU access")
    value.add_argument("--worker", choices=("gpu", "authorize", "download", "metadata", "smoke"), help=argparse.SUPPRESS)
    value.add_argument("--model", choices=tuple(MODELS), help=argparse.SUPPRESS)
    value.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.port <= 65535 or args.min_free_gib < 1 or args.startup_timeout_sec < 1:
        raise ValueError("Invalid port, minimum disk space, or timeout")
    args.bundle = args.bundle.resolve()
    args.work_root = args.work_root.resolve()
    models = selected_models(args.models)
    if args.plan:
        print(json.dumps({"version": VERSION, "action": args.action, "work_root": str(args.work_root),
              "bundle": str(args.bundle), "models": {name: MODELS[name] for name in models},
              "python": PYTHON_VERSION + " (new managed venv)", "vllm": VLLM_VERSION, "torch_backend": "cu129",
              "os_packages": SYSTEM_PACKAGES, "install_driver_if_needed": args.install_driver,
              "steps": ["GCP-only host guard", "bundle/disk/overwrite checks", "OS prerequisites", "driver check",
                        "new isolated environments", "pinned dependencies + pip check + lock", "GPU BF16 matrix check",
                        "weight access check", "download pinned models", "sequential BF16 servers and 2-case live format checks",
                        "READY.json", "full paired inference only with --action run"],
              "changes_made": False}, indent=2))
        return 0
    if args.worker:
        if not args.worker_output:
            raise ValueError("Internal worker needs an output path")
        return worker(args)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}; stopping owned processes")
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        return Bootstrap(args).run()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RebootRequired as exc:
        print(f"REBOOT REQUIRED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(75)
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        raise SystemExit(2)
