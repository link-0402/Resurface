# Shared by generate-blender-repository.ps1 and verify-blender-repository.ps1, so both
# name the package the same way.  The id, name and version come from the add-on's
# blender_manifest.toml; the archive is named after the add-on name, with spaces
# turned into hyphens ("Resurface" -> Resurface.zip, "XIV Instant Edit" ->
# XIV-Instant-Edit.zip).

function Get-ExtensionPackage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath
    )

    $manifest = Get-Content -LiteralPath $ManifestPath -Raw
    # Only top-level keys; they come before the first [section].
    $topLevel = ($manifest -split '(?m)^\[', 2)[0]
    $values = @{}
    foreach ($key in 'id', 'name', 'version') {
        $match = [regex]::Match($topLevel, ('(?m)^{0}\s*=\s*"([^"]+)"' -f $key))
        if (-not $match.Success) {
            throw "No $key found in $ManifestPath"
        }
        $values[$key] = $match.Groups[1].Value
    }

    [pscustomobject]@{
        Id      = $values['id']
        Name    = $values['name']
        Version = $values['version']
        Archive = ($values['name'] -replace '[^A-Za-z0-9._-]+', '-').Trim('-') + '.zip'
    }
}
