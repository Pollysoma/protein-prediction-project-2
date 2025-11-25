#!/usr/bin/env pwsh
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Image = 'nvcr.io/nvidia/clara/bionemo-framework:2.7.1'
$NgcOrg = 'nvidia'

$WorkDir = (Get-Location).ProviderPath
$ModelDirRelative = 'models'
$DataDirRelative = 'outputs_splits_final'
$ModelDir = Join-Path $WorkDir $ModelDirRelative
$DataDir = Join-Path $WorkDir $DataDirRelative

$Models = @(
  [pscustomobject]@{ Size = '8M';   Source = 'ngc';   Ref = 'nvidia/clara/esm2nv8m:2.1';     Tar = 'esm2_8m_checkpoint.tar.gz';  EnvVar = $null }
  [pscustomobject]@{ Size = '650M'; Source = 'ngc';   Ref = 'nvidia/clara/esm2nv650m:2.1';    Tar = 'esm2_650m_checkpoint.tar.gz'; EnvVar = $null }
)

function Log([string]$Message) {
  Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Message"
}

function Get-MicroBatch([string]$Size) {
  switch ($Size) {
    '8M' { return 8 }
    '35M' { return 4 }
    '150M' { return 2 }
    '650M' { return 1 }
    default { return 1 }
  }
}

function Resolve-TarPath([string]$Dest, [string]$TarName, [string]$EnvName) {
  if ($EnvName) {
    $override = [Environment]::GetEnvironmentVariable($EnvName)
    if ($override -and (Test-Path $override)) {
      return (Resolve-Path $override).Path
    }
  }
  $match = Get-ChildItem -Path $Dest -Recurse -File -Filter $TarName -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($match) { return $match.FullName }
  return $null
}

function Download-Model([string]$Size, [string]$Source, [string]$Ref, [string]$TarName, [string]$EnvVar) {
  $dest = Join-Path $ModelDir $Size
  $unpack = Join-Path $dest 'unpacked'
  if (Test-Path $unpack) {
    Log "Model $Size already present at $unpack"
    return
  }

  New-Item -ItemType Directory -Force -Path $dest | Out-Null

  if ($Source -eq 'ngc') {
    if ([string]::IsNullOrWhiteSpace($env:NGC_CLI_API_KEY)) {
      throw "NGC_CLI_API_KEY not set. Needed to fetch $Size ($Ref)."
    }
    Log "Downloading $Ref -> $dest"
    $bashScript = @"
set -euo pipefail
apt-get update >/dev/null && apt-get install -y unzip >/dev/null
curl -fSL https://ngc.nvidia.com/downloads/ngccli_linux.zip -o /tmp/ngccli_linux.zip
unzip -o /tmp/ngccli_linux.zip -d /tmp >/dev/null
chmod +x /tmp/ngc-cli/ngc
export NGC_CLI_ORG=$NgcOrg
export NGC_CLI_ACCEPT_LICENSE=yes
/tmp/ngc-cli/ngc registry model download-version $Ref --dest /host/$ModelDirRelative/$Size
"@

    docker run --rm -e 'NGC_CLI_API_KEY' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
      -v "${WorkDir}:/host" $Image bash -lc "$bashScript"
  }

  if ($Source -eq 'hf') {
    $repo = $Ref
    $repoOverride = Get-Item "env:HF_REPO_$Size" -ErrorAction SilentlyContinue
    if ($repoOverride) { $repo = $repoOverride.Value }
    Log "Downloading $TarName from Hugging Face repo $repo"
    $py = @'
import os, shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id, filename, dest_dir = sys.argv[1:4]
token = os.environ.get("HF_TOKEN") or None
dest = Path(dest_dir)
dest.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(repo_id=repo_id, filename=filename, token=token, local_dir=None)
target = dest / Path(path).name
if path != str(target):
    shutil.copy2(path, target)
print(target)
'@
    $py | python - $repo $TarName $dest
  }

  $tarPath = Resolve-TarPath -Dest $dest -TarName $TarName -EnvName $EnvVar
  if (-not $tarPath) {
    $hint = if ($EnvVar) {
      "Download $TarName from Hugging Face, place it under $dest, or set $EnvVar to an absolute path."
    } else {
      "Download $TarName from Hugging Face and place it under $dest."
    }
    throw "Checkpoint for $Size not found. $hint"
  }

  New-Item -ItemType Directory -Force -Path $unpack | Out-Null
  & tar -xf "$tarPath" -C "$unpack"
  Log "Unpacked $Size to $unpack"
}

function Convert-ParquetToCsv([string]$ParquetPath, [string]$CsvPath) {
  $script = @'
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)
if "sequence" in df.columns and "sequences" not in df.columns:
    df = df.rename(columns={"sequence": "sequences"})
if "sequences" not in df.columns:
    raise SystemExit("Input needs a 'sequences' or 'sequence' column")
if "accession" not in df.columns:
    df["accession"] = df.index.astype(str)
df[["sequences","accession"]].to_csv(dst, index=False)
print(f"wrote {dst} rows={len(df)}")
'@

  $script | python - $ParquetPath $CsvPath
}

function Convert-PredictionsToSafetensors([string]$PredDir, [string]$OutPath, [string]$CsvPath) {
  $script = @'
import sys, json, torch
from pathlib import Path
from safetensors.torch import save_file
import pandas as pd

pred_dir, out_path, csv_path = map(Path, sys.argv[1:4])
pt_files = sorted(pred_dir.glob("predictions__rank_*__dp_rank_*.pt"))
if not pt_files:
    raise SystemExit(f"No predictions__rank_* files in {pred_dir}")
data = torch.load(pt_files[0], map_location="cpu")
emb = data.get("embeddings")
if emb is None:
    raise SystemExit("No embeddings key in predictions file")
df = pd.read_csv(csv_path)
ids = df["accession"].tolist()
if emb.shape[0] != len(ids):
    raise SystemExit(f"Mismatch: embeddings {emb.shape[0]} vs ids {len(ids)}")
meta = {"ids_json": json.dumps(ids)}
save_file({"embeddings": emb}, out_path, metadata=meta)
print(f"saved {out_path} shape {tuple(emb.shape)}")
'@

  $script | python - $PredDir $OutPath $CsvPath
}

function Invoke-Inference([string]$CkptPathRel, [string]$CsvPathRel, [string]$PredDirRel, [int]$MicroBatch) {
  $bashScript = @"
set -euo pipefail
python /workspace/bionemo2/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/infer_esm2.py \
  --checkpoint-path /host/$CkptPathRel \
  --data-path /host/$CsvPathRel \
  --results-path /host/$PredDirRel \
  --include-embeddings \
  --micro-batch-size $MicroBatch \
  --precision bf16-mixed \
  --num-gpus 1 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1
"@

  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
    -v "${WorkDir}:/host" $Image bash -lc "$bashScript"
}

try {
  New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
  if (-not (Test-Path $DataDir)) {
    throw "Data directory not found: $DataDir"
  }

  $parquetFiles = Get-ChildItem -Path $DataDir -Filter '*.parquet' -File | Sort-Object Name
  if (-not $parquetFiles) {
    throw "No parquet files in $DataDir"
  }

  foreach ($model in $Models) {
    try {
      Download-Model -Size $model.Size -Source $model.Source -Ref $model.Ref -TarName $model.Tar -EnvVar $model.EnvVar
    } catch {
      switch ($model.Source) {
        'local' {
          $hintVar = if ($model.EnvVar) { $model.EnvVar } else { 'a local tarball' }
          Write-Warning "Skipping $($model.Size); checkpoint missing. Set $hintVar or place $($model.Tar) under models/$($model.Size)/."
          continue
        }
        'hf' {
          Write-Warning "Skipping $($model.Size); Hugging Face download failed or checkpoint missing. Check HF_TOKEN (if needed) and repo/file availability."
          continue
        }
        default { throw }
      }
    }
    $ckptRel = "$ModelDirRelative/$($model.Size)/unpacked"
    foreach ($pq in $parquetFiles) {
      $base = [IO.Path]::GetFileNameWithoutExtension($pq.Name)
      $csvPath = Join-Path $DataDir "$base.csv"
      $csvRel = "$DataDirRelative/$base.csv"
      if (-not (Test-Path $csvPath)) {
        Convert-ParquetToCsv -ParquetPath $pq.FullName -CsvPath $csvPath
      }

      $predDir = Join-Path $DataDir "${base}_$($model.Size)_pred"
      $predDirRel = "$DataDirRelative/${base}_$($model.Size)_pred"
      $outPath = Join-Path $DataDir "${base}_$($model.Size)_embeddings.safetensors"
      if (Test-Path $outPath) {
        Log "Skipping $($model.Size) on $base; embeddings already exist at $outPath"
        continue
      }
      $mbs = Get-MicroBatch -Size $model.Size
      Log "Running $($model.Size) on $base (micro-batch $mbs)"
      New-Item -ItemType Directory -Force -Path $predDir | Out-Null
      Invoke-Inference -CkptPathRel $ckptRel -CsvPathRel $csvRel -PredDirRel $predDirRel -MicroBatch $mbs
      Convert-PredictionsToSafetensors -PredDir $predDir -OutPath $outPath -CsvPath $csvPath
    }
  }
} catch {
  Write-Error $_
  exit 1
}
