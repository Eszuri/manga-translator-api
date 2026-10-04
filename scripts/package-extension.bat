@echo off
setlocal
set "MANGA_EXTENSION_SCRIPT=%~f0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$source = Get-Content -LiteralPath $env:MANGA_EXTENSION_SCRIPT -Raw; $marker = '# POWERSHELL_START'; $start = $source.LastIndexOf($marker); if ($start -lt 0) { throw 'Embedded extension builder was not found.' }; Invoke-Expression $source.Substring($start + $marker.Length)"
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
if not "%BUILD_EXIT_CODE%"=="0" echo [FAILED] Extension package build failed.
pause
exit /b %BUILD_EXIT_CODE%

# POWERSHELL_START
$ErrorActionPreference = 'Stop'
$scriptDirectory = Split-Path -Parent $env:MANGA_EXTENSION_SCRIPT
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDirectory '..')).Path
$extensionDirectory = Join-Path $projectRoot 'extension'
$manifest = Join-Path $extensionDirectory 'manifest.json'
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Extension manifest not found: $manifest"
}

$distDirectory = Join-Path $projectRoot 'dist'
New-Item -ItemType Directory -Path $distDirectory -Force | Out-Null
$zipOutput = Join-Path $distDirectory 'manga-translator.zip'
$crxOutput = Join-Path $distDirectory 'manga-translator.crx'
$keyOutput = Join-Path $distDirectory 'manga-translator.pem'
$temporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ('manga-extension-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null

try {
    Write-Host 'Building extension package...'
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $temporaryZip = Join-Path $temporaryDirectory 'manga-translator.zip'
    $archive = [System.IO.Compression.ZipFile]::Open($temporaryZip, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        $extensionPrefix = [System.IO.Path]::GetFullPath($extensionDirectory).TrimEnd('\') + '\'
        foreach ($file in Get-ChildItem -LiteralPath $extensionDirectory -Recurse -File) {
            $relativeName = $file.FullName.Substring($extensionPrefix.Length).Replace('\', '/')
            $parts = $relativeName -split '/'
            if ($parts -contains '.git' -or $parts -contains '__pycache__' -or
                $file.Name -in @('.DS_Store', 'Thumbs.db') -or $file.Extension -eq '.pyc') {
                continue
            }
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $file.FullName, $relativeName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    } finally {
        $archive.Dispose()
    }
    Move-Item -LiteralPath $temporaryZip -Destination $zipOutput -Force
    Write-Host "Built: $zipOutput"

    $browserCandidates = @(
        $env:CHROME_PATH,
        'C:\Program Files\Google\Chrome\Application\chrome.exe',
        'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
        (Join-Path $env:LOCALAPPDATA 'Google\Chrome\Application\chrome.exe'),
        'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe'
    )
    $browser = $browserCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
    if (-not $browser) {
        foreach ($name in @('chrome', 'google-chrome', 'chromium', 'msedge', 'brave')) {
            $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($command) { $browser = $command.Source; break }
        }
    }
    if (-not $browser) {
        Write-Host 'No Chromium browser found. Only the ZIP package was built.'
        return
    }

    $stagingExtension = Join-Path $temporaryDirectory 'extension'
    Copy-Item -LiteralPath $extensionDirectory -Destination $stagingExtension -Recurse -Force
    $browserArguments = @('"' + "--pack-extension=$stagingExtension" + '"', '--no-message-box')
    if (Test-Path -LiteralPath $keyOutput -PathType Leaf) {
        $browserArguments += '"' + "--pack-extension-key=$keyOutput" + '"'
    }
    $browserProcess = Start-Process -FilePath $browser -ArgumentList $browserArguments -PassThru -WindowStyle Hidden
    if (-not $browserProcess.WaitForExit(60000)) {
        Stop-Process -Id $browserProcess.Id -Force -ErrorAction SilentlyContinue
        throw 'CRX packaging timed out. The ZIP package was built successfully.'
    }
    $generatedCrx = Join-Path $temporaryDirectory 'extension.crx'
    if ($browserProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $generatedCrx -PathType Leaf)) {
        throw 'CRX packaging failed. The existing CRX was not updated; use the newly built ZIP.'
    }
    $generatedKey = Join-Path $temporaryDirectory 'extension.pem'
    if ((Test-Path -LiteralPath $generatedKey -PathType Leaf) -and
        -not (Test-Path -LiteralPath $keyOutput -PathType Leaf)) {
        Move-Item -LiteralPath $generatedKey -Destination $keyOutput
    }
    Move-Item -LiteralPath $generatedCrx -Destination $crxOutput -Force
    Write-Host "Built: $crxOutput"
} finally {
    $resolvedTemporary = [System.IO.Path]::GetFullPath($temporaryDirectory)
    $temporaryRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if ($resolvedTemporary.StartsWith($temporaryRoot, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedTemporary) -match '^manga-extension-[0-9a-f]{32}$') {
        Remove-Item -LiteralPath $resolvedTemporary -Recurse -Force -ErrorAction SilentlyContinue
    }
}
