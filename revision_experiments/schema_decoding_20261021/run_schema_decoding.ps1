param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('prepare','infer','evaluate','test')]
    [string]$Stage,
    [Parameter(Mandatory=$true)]
    [string]$ProjectRoot,
    [Parameter(Mandatory=$true)]
    [string]$PublicRoot,
    [Parameter(Mandatory=$true)]
    [string]$Python,
    [string]$WslDistribution = 'Ubuntu',
    [string]$WslUser = '',
    [string]$Bundle = '',
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
$project = $ProjectRoot.TrimEnd('/')
$package = "$project/revision_experiments/schema_decoding_20261021"
$public = $PublicRoot.TrimEnd('/')
if ($Bundle -eq '') { $Bundle = "$package/cloud_bundle" }
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
# Use an explicit existing interpreter without modifying its environment.
$bash = @"
set -e
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH='$project'
export VLABENCH_PROJECT_ROOT='$project'
export VLABENCH_PUBLIC_ROOT='$public'
$(Quote-Bash $Python) -u $quoted
"@
$wslArgs = @('-d', $WslDistribution)
if ($WslUser -ne '') { $wslArgs += @('--user', $WslUser) }
& wsl @wslArgs -- bash -lc $bash
exit $LASTEXITCODE
