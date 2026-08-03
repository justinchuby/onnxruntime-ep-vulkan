# Fact Checker — mechanical citation derivation for Morpheus's rule register.
# Reproducible: run from repo root. Emits every hit for every anchor phrase, with
# an owner tag derived from path. No judgement is applied here; judgement is in
# rule-register-derived.md.

$anchors = [ordered]@{
  'P01-pattern-attracts'   = 'attracting cases that do not belong'
  'P02-coverage-composes'  = 'compose to the weaker extent'
  'P02b-coverage-composes' = 'coverage does not compose'
  'P03-convict'            = 'cannot convict'
  'P03b-acquit'            = 'cannot acquit'
  'P04-prohibition-blind'  = 'blind to the state'
  'P05-fault-scope'        = 'scope of what you cannot locate'
  'P06-harden-narrow'      = 'hardened because it is about to pass'
  'P07-frame-not-carried'  = 'comment with a schema'
  'P08-identifying-field'  = 'read a field that identifies something'
  'P09-key-frame-subject'  = 'key is the form'
  'P10-one-hash'           = 'blind to the compiler'
  'P11-fingerprint-fn'     = 'production is a function'
  'P12-ruling-party'       = 'adversarial review by a party with no stake'
  'P13-proven-elsewhere'   = 'PROVEN-ELSEWHERE'
  'P14-frame-vs-absence'   = 'indistinguishable from a key absence'
  'P15-verdict-filename'   = 'names its own frame in its filename'
  'P16-binary-that-ran'    = 'the frame is the binary that ran it'
  'P17-date-is-a-claim'    = 'a date is a claim'
  'P18-cheapest-satisf'    = 'cheapest satisfaction'
  'P19-wrong-and-stable'   = 'wrong \*?and stable'
}

function Get-Owner([string]$p) {
  switch -Regex ($p) {
    '^docs/DESIGN\.md'                 { 'MORPHEUS (self)' ; break }
    '^\.squad/agents/morpheus/'        { 'MORPHEUS (self)' ; break }
    '^\.squad/orchestration-log/.*morpheus' { 'MORPHEUS (self, routing)' ; break }
    '^\.squad/orchestration-log/'      { 'COORDINATOR (in-conversation)' ; break }
    '^\.squad/log/'                    { 'SCRIBE (narration)' ; break }
    '^\.squad/decisions'               { 'DECISION RECORD (check **By:**)' ; break }
    '^\.squad/agents/fact-checker/'    { 'FACT CHECKER' ; break }
    '^\.squad/fact-checker/'           { 'FACT CHECKER' ; break }
    '^\.squad/rai/'                    { 'RAI' ; break }
    '^\.squad/agents/([a-z]+)/'        { "AGENT $($Matches[1])" ; break }
    '^docs/OP_COVERAGE\.md'            { 'MOUSE' ; break }
    '^docs/PERF\.md'                   { 'NIOBE' ; break }
    '^docs/PLATFORMS\.md'              { 'LINK' ; break }
    '^docs/ENGINE\.md'                 { 'SWITCH' ; break }
    '^tests/'                          { 'TRINITY (tests)' ; break }
    '^bench/'                          { 'NIOBE (bench)' ; break }
    '^ci/|^\.github/'                  { 'LINK (ci)' ; break }
    '^rust/src/(ops|registry)'         { 'MOUSE (code)' ; break }
    '^rust/'                           { 'CODE (tank/switch/mouse)' ; break }
    default                            { 'OTHER' }
  }
}

foreach ($k in $anchors.Keys) {
  $pat = $anchors[$k]
  Write-Output "### $k  /$pat/"
  $hits = git grep -n -i -E -- "$pat" HEAD 2>$null
  if (-not $hits) { Write-Output "    (no hits)"; continue }
  foreach ($h in $hits) {
    $rest = $h -replace '^HEAD:', ''
    $path = ($rest -split ':', 2)[0] -replace '\\', '/'
    $ln   = (($rest -split ':', 3)[1])
    Write-Output ("    [{0,-28}] {1}:{2}" -f (Get-Owner $path), $path, $ln)
  }
}
