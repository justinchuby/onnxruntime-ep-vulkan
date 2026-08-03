# Characterises the intermittent `counters::tests` failure reported in round 35.
#
# Two arms, because the two cheap hypotheses predict different things:
#   isolated : `cargo test --lib counters` -- only the 21 counters tests run.
#              A failure here is order-dependence or a race AMONG the counters tests.
#   full     : `cargo test --lib` -- all ~505 tests share the same process and
#              the same libtest thread pool.  A failure here but NOT in `isolated`
#              says the interaction is with a NEIGHBOUR test, not within the module.
#
# Failure text is kept this time (round 35 reported the rate without the message).
param(
    [int]$IsolatedReps = 20,
    [int]$FullReps = 8,
    [string]$OutDir = "$PSScriptRoot\..\..\bench\results\counters_intermittent"
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir "repeats.jsonl"
Remove-Item $log -ErrorAction SilentlyContinue

function Invoke-Arm {
    param([string]$Arm, [string[]]$CargoArgs, [int]$Reps)
    for ($i = 1; $i -le $Reps; $i++) {
        $raw = & cargo @CargoArgs 2>&1 | Out-String
        $failed = ($LASTEXITCODE -ne 0)
        $failLines = ($raw -split "`n" | Where-Object { $_ -match '^(test .*FAILED|failures:|---- .* stdout ----|thread .* panicked)' }) -join ' | '
        $rec = [ordered]@{
            arm       = $Arm
            rep       = $i
            exit_code = $LASTEXITCODE
            failed    = $failed
            summary   = (($raw -split "`n" | Where-Object { $_ -match '^test result:' }) -join ' ; ').Trim()
            fail_text = $failLines
        }
        ($rec | ConvertTo-Json -Compress) | Add-Content $log
        if ($failed) {
            $raw | Set-Content (Join-Path $OutDir "fail-$Arm-$i.txt")
            Write-Host "FAIL $Arm rep $i -- full output kept"
        } else {
            Write-Host "ok   $Arm rep $i"
        }
    }
}

Invoke-Arm -Arm "isolated" -CargoArgs @("test", "--lib", "counters") -Reps $IsolatedReps
Invoke-Arm -Arm "full" -CargoArgs @("test", "--lib") -Reps $FullReps
Write-Host "log: $log"
