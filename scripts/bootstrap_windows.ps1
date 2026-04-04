param(
    [string]$ProjectRoot = ".",
    [ValidateSet("infer-cpu", "infer-cu", "dev")]
    [string]$Profile = "infer-cpu"
)

$ErrorActionPreference = "Stop"

function Assert-UvInstalled {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $uv) {
        Write-Host "uv detected: $($uv.Source)"
        return
    }

    Write-Host "uv not found. Installing with official script..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    $env:Path += ";$HOME\.local\bin;$HOME\.cargo\bin"
    if ($null -eq (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installation failed. Re-open PowerShell and run again."
    }
}

function Resolve-ProjectRoot([string]$PathInput) {
    $resolved = Resolve-Path $PathInput
    Set-Location $resolved
    if (-not (Test-Path "pyproject.toml")) {
        throw "pyproject.toml not found under $resolved"
    }
    return (Get-Location).Path
}

Assert-UvInstalled
$root = Resolve-ProjectRoot -PathInput $ProjectRoot

Write-Host "Project root: $root"
Write-Host "Syncing base and dev dependencies..."
uv sync --extra dev

switch ($Profile) {
    "infer-cpu" {
        Write-Host "Syncing CPU inference dependencies..."
        uv sync --extra infer-cpu
    }
    "infer-cu" {
        Write-Host "Syncing CUDA inference dependencies..."
        uv sync --extra infer-cu
    }
    "dev" {
        Write-Host "Using dev-only profile."
    }
}

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Next commands:"
Write-Host "  uv run python -m dltr data validate"
Write-Host "  uv run python -m dltr data stats"
Write-Host "  uv run python -m dltr demo"
Write-Host "  uv run python -m dltr evaluate end2end"
