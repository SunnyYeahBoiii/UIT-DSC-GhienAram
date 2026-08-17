param(
  [string]$Destination = "$PSScriptRoot\kaggle_data_upload"
)

$ErrorActionPreference = "Stop"
$SourceRoot = "C:\Users\LOQ\Downloads"
$DestinationPath = [System.IO.Path]::GetFullPath($Destination)
if ($DestinationPath.Length -lt 10 -or $DestinationPath -eq [System.IO.Path]::GetPathRoot($DestinationPath)) {
  throw "Destination không an toàn: $DestinationPath"
}

New-Item -ItemType Directory -Force -Path $DestinationPath | Out-Null
Copy-Item -LiteralPath (Join-Path $SourceRoot "train.json") -Destination $DestinationPath
Copy-Item -LiteralPath (Join-Path $SourceRoot "public-official.json") -Destination $DestinationPath
Copy-Item -LiteralPath (Join-Path $SourceRoot "DSC2026_Task2_LegalQA_Data_Overview.docx") -Destination $DestinationPath
Copy-Item -LiteralPath (Join-Path $SourceRoot "selected-contexts") -Destination $DestinationPath -Recurse

$Files = Get-ChildItem -LiteralPath (Join-Path $DestinationPath "selected-contexts") -Filter "context_*.json" -File
if ($Files.Count -ne 8532) {
  throw "Corpus staging không đủ 8.532 file: $($Files.Count)"
}
Write-Host "Sẵn sàng upload thủ công lên Kaggle Dataset: $DestinationPath"

