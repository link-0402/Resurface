[CmdletBinding()]
param(
    [string]$RepositoryPath = '',
    [string]$ExpectedMinimum = '4.2.0'
)

# Checks that index.json describes the Resurface.zip next to it (size and hash) and
# the version in resurface/blender_manifest.toml.  Blender refuses to install a
# package whose size or hash differs from its index entry.

$ErrorActionPreference = 'Stop'

$scriptRoot = if ($PSScriptRoot) {
    $PSScriptRoot
} else {
    Split-Path -Parent $MyInvocation.MyCommand.Definition
}
if ([string]::IsNullOrWhiteSpace($RepositoryPath)) {
    $RepositoryPath = [System.IO.Path]::Combine($scriptRoot, '..', 'blender_repo')
}

$resolvedRepository = (Resolve-Path -LiteralPath $RepositoryPath).Path
$archivePath = Join-Path $resolvedRepository 'Resurface.zip'
$indexPath = Join-Path $resolvedRepository 'index.json'
$manifestPath = [System.IO.Path]::Combine($scriptRoot, '..', 'resurface', 'blender_manifest.toml')
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Blender extension package not found: $archivePath"
}
if (-not (Test-Path -LiteralPath $indexPath -PathType Leaf)) {
    throw "Blender repository index not found: $indexPath"
}

$entry = (Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json).data |
    Where-Object { $_.id -eq 'resurface' } |
    Select-Object -First 1
if ($null -eq $entry) {
    throw 'Blender repository index does not contain resurface.'
}
if ($entry.blender_version_min -ne $ExpectedMinimum) {
    throw "Blender minimum mismatch: expected $ExpectedMinimum, found $($entry.blender_version_min)."
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw
$versionMatch = [regex]::Match($manifest, '(?m)^version\s*=\s*"([^"]+)"')
if (-not $versionMatch.Success) {
    throw "No version found in $manifestPath"
}
if ($entry.version -ne $versionMatch.Groups[1].Value) {
    throw "Version mismatch: manifest=$($versionMatch.Groups[1].Value), index=$($entry.version). Regenerate the repository."
}

$archive = Get-Item -LiteralPath $archivePath
if ([int64]$entry.archive_size -ne $archive.Length) {
    throw "Blender archive size mismatch: index=$($entry.archive_size), actual=$($archive.Length)."
}

$actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if (([string]$entry.archive_hash).ToLowerInvariant() -ne "sha256:$actualHash") {
    throw "Blender archive hash mismatch: index=$($entry.archive_hash), actual=sha256:$actualHash."
}

Write-Host "Verified Blender repository: Resurface $($entry.version), $($archive.Length) bytes, sha256:$actualHash"
