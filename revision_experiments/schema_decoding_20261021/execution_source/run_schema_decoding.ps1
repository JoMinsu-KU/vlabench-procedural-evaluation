param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('prepare','infer','evaluate','test')]
    [string]$Stage,
    [string]$Bundle = '/mnt/c/VLABenchProcEval/revision_experiments/schema_decoding_20261021/cloud_bundle',
    [string]$OutputRoot = '',
    [string]$GenerationRoot = '',
    [string]$Model = '',
    [string]$ServingConfig = '',
    [string]$BaseUrl = 'http://127.0.0.1:8000/v1',
    [ValidateSet('both','json_object','json_schema')]
    [string]$Condition = 'both',
    [int]$MaxCases = 0,
    [int]$PerTask = 10,
    [switch]$ValidationOnly,
    [switch]$Resume,
    [switch]$RetryApiErrors
)
$ErrorActionPreference = 'Stop'
$project = '/mnt/c/VLABenchProcEval'
$package = "$project/revision_experiments/schema_decoding_20261021"
$public = '/mnt/c/SILVER/data/public_view/vlabench_vlm4500_public'
function Quote-Bash([string]$Value) {
    if ($Value.Contains("'")) { throw 'A single quote in an argument is not supported by this wrapper.' }
    return "'$Value'"
}
if ($Stage -eq 'test') {
    $argv = @('-B', "$package/test_schema_decoding.py")
} else {
    if ($OutputRoot -eq '') { throw 'OutputRoot must explicitly name a NEW directory.' }
    $argv = @('-B', "$package/schema_decoding.py", $Stage, '--output-root', $OutputRoot)
    switch ($Stage) {
        'prepare' {
            $argv += @('--project-root', $project, '--public-root', $public,
                '--manifest', "$project/datasets/manifests/matched_2133_symbolic_procedural_cases.json", '--per-task', [string]$PerTask)
        }
        'infer' {
            if ($Model -eq '' -or $ServingConfig -eq '') { throw 'Model and ServingConfig are required for inference.' }
            $argv += @('--bundle', $Bundle, '--model', $Model, '--serving-config', $ServingConfig, '--base-url', $BaseUrl, '--condition', $Condition)
            if ($RetryApiErrors) { $argv += '--retry-api-errors' }
        }
        'evaluate' {
            if ($GenerationRoot -eq '') { throw 'GenerationRoot is required for offline evaluation.' }
            $argv += @('--project-root', $project, '--public-root', $public, '--bundle', $Bundle, '--generation-root', $GenerationRoot)
            if ($ValidationOnly) { $argv += '--validation-only' }
        }
    }
    if ($MaxCases -gt 0) { $argv += @('--max-cases', [string]$MaxCases) }
    if ($Resume) { $argv += '--resume' }
}
$quoted = ($argv | ForEach-Object { Quote-Bash $_ }) -join ' '
# Activate the existing simulator environment; do not install or upgrade any package.
$bash = @"
set -e
source /home/jms/miniforge3/etc/profile.d/conda.sh
conda activate vlabench310_repaired
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='$project'
python -u $quoted
"@
& wsl -d Ubuntu-vlabench --user jms -- bash -lc $bash
exit $LASTEXITCODE
