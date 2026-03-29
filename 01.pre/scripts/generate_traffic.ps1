#Requires -Version 5.1
param(
    [string]$LiteLLMHost = "localhost",
    [int]   $Port         = 4000,
    [int]   $Rounds       = 3,
    [double]$Delay        = 1.0,
    [string]$ApiKey       = "sk-1234",
    [string[]]$Models   = @("qwen3:0.6b","deepseek-r1","gpt-oss","gpt-5.2-chat","grok-3-mini","Phi-4")
)


$ErrorActionPreference = "SilentlyContinue"
$BaseUrl = "http://${LiteLLMHost}:${Port}"
$script:HitIDs    = [System.Collections.Generic.List[string]]::new()
$script:MissedIDs = [System.Collections.Generic.List[string]]::new()
$script:AllIDs    = [System.Collections.Generic.List[string]]::new()  # global, never reset
$AuthHeaders = @{
    "Authorization" = "Bearer $ApiKey"
    "Content-Type"  = "application/json"
}

# ── Stats ─────────────────────────────────────────────────────────────────────
$Stats = @{
    Total          = 0
    Success        = 0
    Failed         = 0
    CacheHits      = 0
    TotalTokens    = 0
    TotalLatencyMs = 0
    Errors         = [System.Collections.Generic.List[string]]::new()
}

# ── Colours ───────────────────────────────────────────────────────────────────
function Write-Cache  { param($msg) Write-Host "  " -NoNewline; Write-Host "[CACHE HIT ]" -ForegroundColor Green -NoNewline; Write-Host " $msg" }
function Write-Miss   { param($msg) Write-Host "  " -NoNewline; Write-Host "[CACHE MISS]" -ForegroundColor DarkYellow   -NoNewline; Write-Host " $msg" }
function Write-Fail   { param($msg) Write-Host "  " -NoNewline; Write-Host "[FAIL      ]" -ForegroundColor Red     -NoNewline; Write-Host " $msg" }
function Write-Warn   { param($msg) Write-Host "  " -NoNewline; Write-Host "[WARN      ]" -ForegroundColor Yellow  -NoNewline; Write-Host " $msg" }
function Write-Info   { param($msg) Write-Host "  " -NoNewline; Write-Host "[INFO      ]" -ForegroundColor Cyan    -NoNewline; Write-Host " $msg" }
function Write-Phase  { param($msg) Write-Host ""; Write-Host "  -- $msg --" -ForegroundColor Cyan; Write-Host "" }

function Get-Timestamp { return (Get-Date).ToString("HH:mm:ss") }

# ── Prompt sets ───────────────────────────────────────────────────────────────
# REPEATED prompts  → 1st round: cache MISS (LLM call)
#                   → later rounds: cache HIT (Redis, ~10ms)
# UNIQUE prompts    → always cache MISS (random seed makes them unique)

$RepeatedPrompts = @(
    @{ role = "user"; content = "In one word, what is 2 + 2?" },
    @{ role = "user"; content = "In one word, what is the capital of France?" },
    @{ role = "user"; content = "In one word, what colour is the sky on a clear day?" }
)

$UniquePrompts = @(
    @{ role = "user"; content = "Generate a random number between 1-1000 and explain it. Seed: $(Get-Random -Minimum 1 -Maximum 9999)" },
    @{ role = "user"; content = "IN single line provide current timestamp philosophy: $((Get-Date).ToString('o'))" },
    @{ role = "user"; content = "Write a random joke in single line$(Get-Random -Minimum 1 -Maximum 100)." }
)

# ── Core request function ─────────────────────────────────────────────────────
function Send-LLMRequest {
    param(
        [array] $Messages,
        [string]$Label
    )

    $Stats.Total++

    $Body1 = @{
        model      = $Model
        messages   = $Messages
        max_tokens = 50
        stream     = $false
        think      = $false
    } | ConvertTo-Json -Depth 5

    $Body2 = @{
        model      = $Model
        messages   = $Messages
        max_tokens = 50
        stream     = $false
    } | ConvertTo-Json -Depth 5

    $StartMs = [int](Get-Date -UFormat %s%3N)  # milliseconds
    $StartTime = Get-Date
    If ($Model -match "gpt-5.2-chat|o3-mini") {
        $Body = $Body2
        $ProcessUniquePrompts = $false
    } else {
        $Body = $Body1
        $ProcessUniquePrompts = $true
    }
    try {
        $Response = Invoke-WebRequest `
            -Uri     "$BaseUrl/v1/chat/completions" `
            -Method  POST `
            -Body    $Body `
            -Headers $AuthHeaders `
            -TimeoutSec 90 `
            -UseBasicParsing `
            -ErrorAction Stop

        $LatencyMs = [int]((Get-Date) - $StartTime).TotalMilliseconds

        $Data       = $Response.Content | ConvertFrom-Json
        $RawContent = $Data.choices[0].message.content
        $Reply      = if ($null -ne $RawContent -and $RawContent -ne "") { $RawContent.Trim() } else { "(empty response)" }
        $Tokens     = if ($Data.usage.total_tokens) { $Data.usage.total_tokens } else { 0 }
        $ResponseId = $Data.id

        # Cache detection: if ID already seen in HitIDs → Cache Hit, else → Cache Miss
        $CacheHit = $script:HitIDs.Contains($ResponseId)
        if (-not $CacheHit) {
            $script:HitIDs.Add($ResponseId)
            $script:MissedIDs.Add($ResponseId)
            $script:AllIDs.Add($ResponseId)   # global, persists across all models
        }

        $Stats.Success++
        $Stats.TotalTokens    += $Tokens
        $Stats.TotalLatencyMs += $LatencyMs
        if ($CacheHit) { $Stats.CacheHits++ }

        $Snippet = if ($Reply.Length -gt 60) { $Reply.Substring(0, 60) + "..." } else { $Reply }
        $Ts      = Get-Timestamp

        if ($CacheHit) {
            Write-Cache "[$Ts] $Label → `"$Snippet`" (${LatencyMs}ms, $Tokens tokens)"
        } else {
            Write-Miss "[$Ts] $Label → `"$Snippet`" (${LatencyMs}ms, $Tokens tokens)"
        }

        return @{ Ok = $true; LatencyMs = $LatencyMs; Tokens = $Tokens; CacheHit = $CacheHit; Reply = $Reply }

    } catch [System.Net.WebException] {
        $LatencyMs = [int]((Get-Date) - $StartTime).TotalMilliseconds
        $StatusCode = [int]$_.Exception.Response.StatusCode
        $ErrorMsg   = "HTTP ${StatusCode}: $($_.Exception.Message)"

        if ($_.Exception.Message -match "timed out|timeout") {
            Write-Warn "[$(Get-Timestamp)] Timeout for '$Label' (>90s) — LLM may still be loading model"
            $Stats.Errors.Add("Timeout: $Label")
        } else {
            Write-Fail "[$(Get-Timestamp)] $Label → $ErrorMsg"
            $Stats.Errors.Add($ErrorMsg)
        }
        $Stats.Failed++
        return @{ Ok = $false }

    } catch {
        $Stats.Failed++
        $ErrorMsg = $_.Exception.Message
        Write-Fail "[$(Get-Timestamp)] $Label → $ErrorMsg"
        $Stats.Errors.Add($ErrorMsg)
        return @{ Ok = $false }
    }
}


# ── Summary table ─────────────────────────────────────────────────────────────
function Show-Summary {
    $AvgLatency  = if ($Stats.Success -gt 0) { [int]($Stats.TotalLatencyMs / $Stats.Success) } else { 0 }
    $CacheRate   = if ($Stats.Success -gt 0) { [math]::Round(($Stats.CacheHits / $Stats.Success) * 100, 0) } else { 0 }
    $LLMCalls = $Stats.Success - $Stats.CacheHits

    Write-Host ""
    Write-Host "  +---------------------------------------+----------+" -ForegroundColor DarkGray
    Write-Host "  | LiteLLM - Traffic Generator Summary              |" -ForegroundColor DarkGray
    Write-Host "  +---------------------------------------+----------+" -ForegroundColor DarkGray

    $rows = @(
        @{ Label = "Total Requests Sent";     Value = "$($Stats.Total)" }
        @{ Label = "Successful";              Value = "$($Stats.Success)" }
        @{ Label = "Failed";                  Value = "$($Stats.Failed)" }
        @{ Label = "Cache Misses";            Value = "$LLMCalls" }
        @{ Label = "Cache Hits";              Value = "$($Stats.CacheHits) ($CacheRate%)" }
        # @{ Label = "Total Cached IDs";      Value = "$($script:AllIDs.Count)" }
        # @{ Label = "Total Hit IDs";         Value = "$($script:HitIDs.Count)" }
        # @{ Label = "Total Missed IDs";      Value = "$($script:MissedIDs.Count)" }
        @{ Label = "Total Tokens used";       Value = "$($Stats.TotalTokens)" }
        @{ Label = "Avg latency (all)";       Value = "${AvgLatency}ms" }
    )

    foreach ($row in $rows) {
        $PaddedLabel = $row.Label.PadRight(37)
        $PaddedVal   = $row.Value.PadLeft(8)
        Write-Host "  | $PaddedLabel | $PaddedVal |" -ForegroundColor DarkGray
    }

    Write-Host "  +---------------------------------------+----------+" -ForegroundColor DarkGray

    if ($Stats.Errors.Count -gt 0) {
        Write-Host ""
        Write-Host "  Errors encountered:" -ForegroundColor Red
        $Stats.Errors | Select-Object -First 5 | ForEach-Object {
            Write-Host "    FAIL $_" -ForegroundColor Red
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "  +======================================================+" -ForegroundColor Cyan
Write-Host "  |   LiteLLM - Traffic Generator                        |" -ForegroundColor Cyan
Write-Host "  +======================================================+" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Target  : $BaseUrl"
Write-Host "  Models  : $($Models -join ', ')"
Write-Host "  Rounds  : $Rounds"
Write-Host "  Delay   : ${Delay}s between requests"
Write-Host ""
Write-Host "  Legend:"
Write-Host "    " -NoNewline; Write-Host "[CACHE MISS]" -ForegroundColor DarkYellow -NoNewline; Write-Host " → real inference call (slow, ~2-8s)"
Write-Host "    " -NoNewline; Write-Host "[CACHE HIT ]" -ForegroundColor Green -NoNewline; Write-Host " → served from Redis   (fast, ~10ms)"
Write-Host ""

# ── Loop over models: run the same traffic sequence per model ────────────────
foreach ($Model in $Models) {
    Write-Phase "Running traffic for model: $Model"

    # reset per-model stats
    $Stats = @{
        Total          = 0
        Success        = 0
        Failed         = 0
        CacheHits      = 0
        TotalTokens    = 0
        TotalLatencyMs = 0
        Errors         = [System.Collections.Generic.List[string]]::new()
    }
    $script:HitIDs    = [System.Collections.Generic.List[string]]::new()
    $script:MissedIDs = [System.Collections.Generic.List[string]]::new()
    # $script:AllIDs is NOT reset here — it accumulates across all models

    Write-Host "  Target  : $BaseUrl"
    Write-Host "  Model   : $Model"
    Write-Host "  Rounds  : $Rounds"
    Write-Host "  Delay   : ${Delay}s between requests"

    # Phase 1: First-time requests (cache MISSes)
    Write-Phase "Phase 1: First-time requests (cache MISSes → real LLM calls)"
    for ($i = 0; $i -lt $RepeatedPrompts.Count; $i++) {
        Send-LLMRequest -Messages @($RepeatedPrompts[$i]) -Label "repeated-prompt-$($i+1) [round 1]" | Out-Null
        Start-Sleep -Milliseconds ($Delay * 1000)
    }

    # Phase 2: Unique/dynamic prompts
    Write-Phase "Phase 2: Unique/dynamic prompts (always cache MISSes)"
    for ($i = 0; $i -lt $UniquePrompts.Count; $i++) {
        Send-LLMRequest -Messages @($UniquePrompts[$i]) -Label "unique-prompt-$($i+1)" | Out-Null
        Start-Sleep -Milliseconds ($Delay * 1000)
    }

    # Phase 3: Repeat prompts in subsequent rounds — expect cache HITs
    for ($round = 2; $round -le $Rounds; $round++) {
        Write-Phase "Phase 3.$($round-1): Round $round — repeated prompts (cache HITs from Redis)"
        for ($i = 0; $i -lt $RepeatedPrompts.Count; $i++) {
            Send-LLMRequest -Messages @($RepeatedPrompts[$i]) -Label "repeated-prompt-$($i+1) [round $round]" | Out-Null
            Start-Sleep -Milliseconds ($Delay * 1000)
        }
    }

    # Show per-model results and Redis keys
    Show-Summary
}

# ── Final ID Report (all models) ─────────────────────────────────────────────
# Write-Host ""
# Write-Host "  +======================================================+" -ForegroundColor Cyan
# Write-Host "  |   Final ID Report (across all models)                |" -ForegroundColor Cyan
# Write-Host "  +======================================================+" -ForegroundColor Cyan

# Write-Host ""
# Write-Host "  AllIDs (global — all cache misses across every model):" -ForegroundColor DarkGray
# Write-Host "  Count: $($script:AllIDs.Count)" -ForegroundColor DarkGray
# if ($script:AllIDs.Count -gt 0) {
#     $script:AllIDs | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
# } else {
#     Write-Host "    (none)" -ForegroundColor DarkGray
# }