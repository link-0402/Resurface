[CmdletBinding()]
param(
    [string]$RepositoryPath = '',
    [string]$ExpectedMinimum = '4.2.0'
)

# Checks that index.json lists only the add-on, under the archive name, id and
# version from resurface/blender_manifest.toml, and that it describes the zip next
# to it (size and hash).  Blender refuses to install a package whose size or hash
# differs from its index entry.

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'extension-package.ps1')

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RepositoryPath)) {
    $RepositoryPath = Join-Path $repoRoot 'blender_repo'
}

$package = Get-ExtensionPackage (Join-Path (Join-Path $repoRoot 'resurface') 'blender_manifest.toml')
$resolvedRepository = (Resolve-Path -LiteralPath $RepositoryPath).Path
$archivePath = Join-Path $resolvedRepository $package.Archive
$indexPath = Join-Path $resolvedRepository 'index.json'
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Blender extension package not found: $archivePath. Regenerate the repository."
}
if (-not (Test-Path -LiteralPath $indexPath -PathType Leaf)) {
    throw "Blender repository index not found: $indexPath"
}

# A second entry means a zip from an earlier build or add-on name is still there.
$entries = @((Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json).data)
if ($entries.Count -ne 1 -or $entries[0].id -ne $package.Id) {
    $listed = ($entries | ForEach-Object { "$($_.id) ($($_.archive_url))" }) -join ', '
    throw "Blender repository index should list only $($package.Id), found: $listed. Regenerate the repository."
}
$entry = $entries[0]
if ($entry.archive_url -ne "./$($package.Archive)") {
    throw "Archive name mismatch: expected ./$($package.Archive), index=$($entry.archive_url). Regenerate the repository."
}
if ($entry.blender_version_min -ne $ExpectedMinimum) {
    throw "Blender minimum mismatch: expected $ExpectedMinimum, found $($entry.blender_version_min)."
}
if ($entry.version -ne $package.Version) {
    throw "Version mismatch: manifest=$($package.Version), index=$($entry.version). Regenerate the repository."
}

$archive = Get-Item -LiteralPath $archivePath
if ([int64]$entry.archive_size -ne $archive.Length) {
    throw "Blender archive size mismatch: index=$($entry.archive_size), actual=$($archive.Length)."
}

$actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if (([string]$entry.archive_hash).ToLowerInvariant() -ne "sha256:$actualHash") {
    throw "Blender archive hash mismatch: index=$($entry.archive_hash), actual=sha256:$actualHash."
}

Write-Host "Verified Blender repository: $($package.Archive), $($package.Name) $($entry.version), $($archive.Length) bytes, sha256:$actualHash"
