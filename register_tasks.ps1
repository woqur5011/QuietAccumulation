# QuietAccumulation Windows 작업 스케줄러 일괄 등록
# 관리자 PowerShell에서 실행하거나, register_tasks.bat을 더블클릭하세요.

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$PY  = "C:\Users\bsjang\AppData\Local\Programs\Python\Python311\python.exe"
$DIR = "C:\Users\bsjang\NEXON_Copilot\QuietAccumulation"

# ── 1) 장중 1시간 watchlist 수집 (평일 09~15시 매 정각) ──────────────────
$action   = New-ScheduledTaskAction -Execute $PY -Argument "$DIR\collector.py" -WorkingDirectory $DIR
$triggers = 9..15 | ForEach-Object {
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "$($_):00"
}
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1) -StartWhenAvailable
Register-ScheduledTask -TaskName "QuietAccumulation_Hourly" `
    -Action $action -Trigger $triggers -Settings $settings -RunLevel Highest -Force
Write-Host "[OK] QuietAccumulation_Hourly 등록 (평일 09~15시 매 정각, watchlist)"

# ── 2) 점심 전종목 수집 (평일 12:00) ──────────────────────────────────────
$action2   = New-ScheduledTaskAction -Execute $PY -Argument "$DIR\collector.py --full" -WorkingDirectory $DIR
$trigger2  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "12:00"
$settings2 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2) -StartWhenAvailable
Register-ScheduledTask -TaskName "QuietAccumulation_Noon" `
    -Action $action2 -Trigger $trigger2 -Settings $settings2 -RunLevel Highest -Force
Write-Host "[OK] QuietAccumulation_Noon 등록 (평일 12:00, full)"

# ── 3) 장마감 전종목 수집 (평일 16:00) ────────────────────────────────────
$action3   = New-ScheduledTaskAction -Execute $PY -Argument "$DIR\collector.py --full" -WorkingDirectory $DIR
$trigger3  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "16:00"
$settings3 = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2) -StartWhenAvailable
Register-ScheduledTask -TaskName "QuietAccumulation_Daily" `
    -Action $action3 -Trigger $trigger3 -Settings $settings3 -RunLevel Highest -Force
Write-Host "[OK] QuietAccumulation_Daily 등록 (평일 16:00, full)"

# ── 결과 확인 ─────────────────────────────────────────────────────────────
Write-Host ""
Get-ScheduledTask -TaskName "QuietAccumulation*" | Select-Object TaskName, State,
    @{n='NextRun';e={(Get-ScheduledTaskInfo $_.TaskName).NextRunTime}}

Write-Host ""
Write-Host "모든 태스크 등록 완료!"
Pause
