$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$qaBrainProcess = $null
try {
    $qaBrainProcess = Start-Process -FilePath "$taskRoot\.venv\Scripts\python.exe" -ArgumentList '-u droffel_sim.py --port 9877 --port-file logs/qa_port.txt --status-file logs/qa_status.json' -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$taskRoot\logs\qa_brain.log" -RedirectStandardError "$taskRoot\logs\qa_brain_error.log"
    $qaRenderProcess = Start-Process -FilePath "$taskRoot\.tools\godot\Godot_v4.7.2-stable_win64_console.exe" -ArgumentList '--path terrarium --script res://scripts/qa.gd -- --sim-port=9877' -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$taskRoot\logs\qa_run.log" -RedirectStandardError "$taskRoot\logs\qa_render_error.log"
    if (-not $qaRenderProcess.WaitForExit(120000)) {
        & taskkill /PID $qaRenderProcess.Id /T /F | Out-Null
        throw 'Godot integration check timed out; see logs/qa_render_error.log'
    }
    $qaExitCode = $qaRenderProcess.ExitCode
    Get-Content -LiteralPath "$taskRoot\logs\qa_run.log"
    Get-Content -LiteralPath "$taskRoot\logs\qa_render_error.log"
} finally {
    if ($qaBrainProcess -and -not $qaBrainProcess.WaitForExit(1500)) {
        & taskkill /PID $qaBrainProcess.Id /T /F | Out-Null
    }
}
exit $qaExitCode
