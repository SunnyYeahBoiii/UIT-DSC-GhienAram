param(
  [Parameter(Mandatory = $true)][string]$Corpus,
  [Parameter(Mandatory = $true)][string]$Questions,
  [string]$OutputZip = "submission.zip"
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $Project "configs/p0.yaml"
$Predictions = Join-Path $Project "artifacts/p0/predictions.json"
$env:PYTHONPATH = Join-Path $Project "src"
$env:PYTHONUTF8 = "1"

function Assert-LastExitCode {
  if ($LASTEXITCODE -ne 0) {
    throw "Lệnh Python thất bại với exit code $LASTEXITCODE"
  }
}

python -m legalqa --config $Config build-index --corpus $Corpus
Assert-LastExitCode
python -m legalqa --config $Config run --questions $Questions --output $Predictions
Assert-LastExitCode
python -m legalqa --config $Config make-submission `
  --questions $Questions --predictions $Predictions --output $OutputZip
Assert-LastExitCode
python -m legalqa --config $Config validate-submission --zip $OutputZip --questions $Questions
Assert-LastExitCode
