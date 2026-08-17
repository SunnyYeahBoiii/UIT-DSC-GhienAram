$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $Project "src"
$env:PYTHONUTF8 = "1"

function Assert-LastExitCode {
  if ($LASTEXITCODE -ne 0) {
    throw "Lệnh Python thất bại với exit code $LASTEXITCODE"
  }
}

python -m legalqa --config (Join-Path $Project "configs/smoke.yaml") prepare-warmup
Assert-LastExitCode
python -m legalqa --config (Join-Path $Project "configs/smoke.yaml") build-index --sparse-only
Assert-LastExitCode
python -m legalqa --config (Join-Path $Project "configs/smoke.yaml") run `
  --questions (Join-Path $Project "artifacts/smoke/data/dev_questions.json") `
  --output (Join-Path $Project "artifacts/smoke/dev_predictions.json")
Assert-LastExitCode
python -m legalqa --config (Join-Path $Project "configs/smoke.yaml") evaluate `
  --predictions (Join-Path $Project "artifacts/smoke/dev_predictions.json") `
  --references (Join-Path $Project "artifacts/smoke/data/dev_references.json") `
  --output (Join-Path $Project "artifacts/smoke/dev_metrics.json")
Assert-LastExitCode
python -m legalqa --config (Join-Path $Project "configs/smoke.yaml") make-submission `
  --questions (Join-Path $Project "artifacts/smoke/data/dev_questions.json") `
  --predictions (Join-Path $Project "artifacts/smoke/dev_predictions.json") `
  --output (Join-Path $Project "artifacts/smoke/submission.zip")
Assert-LastExitCode
