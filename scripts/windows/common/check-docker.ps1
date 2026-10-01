# Fail before any build if Docker Desktop's engine is unavailable.
$ErrorActionPreference = 'Stop'
try {
    $dockerCommand = (Get-Command docker -ErrorAction Stop).Source
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $dockerCommand
    $startInfo.Arguments = 'info --format {{.ServerVersion}}'
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $check = [System.Diagnostics.Process]::Start($startInfo)
    if (-not $check.WaitForExit(15000)) {
        $check.Kill()
        Write-Host '[FAILED] Docker Engine did not respond within 15 seconds. Start or restart Docker Desktop.'
        exit 1
    }
    if ($check.ExitCode -ne 0) {
        Write-Host '[FAILED] Docker Engine is unavailable. Open Docker Desktop and wait until it is running.'
        exit 1
    }
    exit 0
} catch {
    Write-Host "[FAILED] Docker preflight: $($_.Exception.Message)"
    exit 1
}
