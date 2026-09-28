[CmdletBinding()]
param(
    [string]$RepositoryPath = '',
    [string]$BlenderPath = ''
)

# Builds the extension package (named after the add-on, e.g. Resurface.zip) and the
# Blender remote repository listing (index.json, index.html) from it.  Blender reads
# index.json when the repository is added under Preferences > Get Extensions >
# Repositories.

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'extension-package.ps1')

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RepositoryPath)) {
    $RepositoryPath = Join-Path $repoRoot 'blender_repo'
}

$addonRoot = Join-Path $repoRoot 'resurface'
$package = Get-ExtensionPackage (Join-Path $addonRoot 'blender_manifest.toml')
$resolvedRepository = [System.IO.Path]::GetFullPath($RepositoryPath)
$packagePath = Join-Path $resolvedRepository $package.Archive

# Blender: -BlenderPath, else blender on PATH, else the newest one in Program Files.
$blenderExecutable = $null
if (-not [string]::IsNullOrWhiteSpace($BlenderPath)) {
    $resolvedBlenderPath = [System.IO.Path]::GetFullPath($BlenderPath)
    if (-not (Test-Path -LiteralPath $resolvedBlenderPath -PathType Leaf)) {
        throw "Blender executable not found: $resolvedBlenderPath"
    }
    $blenderExecutable = $resolvedBlenderPath
} else {
    $command = Get-Command blender -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        $blenderExecutable = $command.Source
    } else {
        $installed = Get-ChildItem -Path 'C:\Program Files\Blender Foundation\Blender *\blender.exe' -ErrorAction SilentlyContinue |
            Sort-Object { try { [version]($_.Directory.Name -replace '^Blender ', '') } catch { [version]'0.0' } } -Descending |
            Select-Object -First 1
        if ($null -ne $installed) {
            $blenderExecutable = $installed.FullName
        }
    }
}
if ($null -eq $blenderExecutable) {
    throw 'Blender 4.2 or newer is required on PATH, or pass -BlenderPath.'
}
Write-Host "Using Blender: $blenderExecutable"

# server-generate lists every zip in the folder, so clear out the previous package,
# including one left over from an earlier add-on name.
New-Item -ItemType Directory -Path $resolvedRepository -Force | Out-Null
Get-ChildItem -LiteralPath $resolvedRepository -Filter '*.zip' -File | ForEach-Object {
    if ($_.Name -ne $package.Archive) {
        Write-Host "Removing old package: $($_.Name)"
    }
    Remove-Item -LiteralPath $_.FullName -Force
}

# The manifest's [build] section decides which files go into the package.
& $blenderExecutable --background --factory-startup --command extension build "--source-dir=$addonRoot" "--output-filepath=$packagePath"
if ($LASTEXITCODE -ne 0) {
    throw "Blender extension build failed with exit code $LASTEXITCODE."
}
if (-not (Test-Path -LiteralPath $packagePath -PathType Leaf)) {
    throw "Blender package creation did not produce the expected archive: $packagePath"
}

& $blenderExecutable --background --factory-startup --command extension server-generate "--repo-dir=$resolvedRepository" --html
if ($LASTEXITCODE -ne 0) {
    throw "Blender extension repository generation failed with exit code $LASTEXITCODE."
}

Write-Host "Generated Blender extension repository at ${resolvedRepository}: $($package.Archive) ($($package.Name) $($package.Version))"
