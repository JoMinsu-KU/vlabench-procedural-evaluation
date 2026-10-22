#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' 'Python 3 is required. Use an Ubuntu 22.04/24.04 GCP image.' >&2
    exit 2
fi

# One entry point; the controller creates and uses its own Python environments.
exec python3 -u "$SCRIPT_DIR/gcp_bootstrap.py" "$@"
