<#
    add_streamlit_to_path.ps1
    -------------------------
    Adds the per-user Python "Scripts" folder - the place where
    `pip install --user` drops `streamlit.exe` - to the *user* PATH, so the plain
    command `streamlit run app.py` starts working.

    No administrator rights are required, and nothing outside your user profile
    is touched.  Open a NEW terminal afterwards for the change to take effect.

    Usage
    -----
        powershell -ExecutionPolicy Bypass -File .\scripts\add_streamlit_to_path.ps1
        .\scripts\add_streamlit_to_path.ps1 -DryRun     # only show what would change
        .\scripts\add_streamlit_to_path.ps1 -Remove     # undo
#>
[CmdletBinding()]
param(
    [switch]$Remove,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# ------------------------------------------------------------------ #
# 1. Which folders could contain the console scripts?
# ------------------------------------------------------------------ #
$candidates = New-Object System.Collections.Generic.List[string]

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    $userScripts = & $python.Source -c "import site, os; print(os.path.join(site.USER_BASE, 'Scripts'))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $userScripts) {
        $candidates.Add($userScripts.Trim())
    }
}

# Every per-user Python installed on this machine (Python310, Python311, ...).
$pythonRoot = Join-Path $env:APPDATA 'Python'
if (Test-Path $pythonRoot) {
    foreach ($dir in Get-ChildItem $pythonRoot -Directory -ErrorAction SilentlyContinue) {
        $candidates.Add((Join-Path $dir.FullName 'Scripts'))
    }
}

$folders = @($candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)
if ($folders.Count -eq 0) {
    Write-Warning 'No per-user Python Scripts folder was found.'
    Write-Host    'Nothing to do - launch the dashboard with:  python -m streamlit run app.py'
    exit 1
}

foreach ($folder in $folders) {
    $exe = Join-Path $folder 'streamlit.exe'
    $note = if (Test-Path $exe) { 'streamlit.exe found' } else { 'no streamlit.exe here yet' }
    Write-Host ("[found] {0}  ({1})" -f $folder, $note)
}

# ------------------------------------------------------------------ #
# 2. Read the current *user* PATH (admin rights not required)
# ------------------------------------------------------------------ #
$current = [Environment]::GetEnvironmentVariable('Path', 'User')
if ([string]::IsNullOrWhiteSpace($current)) { $current = '' }
$entries = @($current -split ';' | Where-Object { $_ -ne '' })

if ($Remove) {
    $kept = @($entries | Where-Object { $folders -notcontains $_ })
    if ($kept.Count -eq $entries.Count) {
        Write-Host 'Nothing to remove - those folders were not on the user PATH.'
        exit 0
    }
    if ($DryRun) {
        Write-Host ('[dry-run] would remove: ' + ($folders -join ', '))
        exit 0
    }
    [Environment]::SetEnvironmentVariable('Path', ($kept -join ';'), 'User')
    Write-Host ('Removed from the user PATH: ' + ($folders -join ', '))
    Write-Host 'Open a new terminal for the change to take effect.'
    exit 0
}

$added = @($folders | Where-Object { $entries -notcontains $_ })
if ($added.Count -eq 0) {
    Write-Host 'Already on the user PATH - `streamlit` should work in a new terminal.'
    Write-Host ('Current user PATH: ' + $current)
    exit 0
}

if ($DryRun) {
    Write-Host ('[dry-run] would add to the user PATH: ' + ($added -join ', '))
    Write-Host ('Current user PATH: ' + $current)
    exit 0
}

[Environment]::SetEnvironmentVariable('Path', (($entries + $added) -join ';'), 'User')
$env:Path = $env:Path.TrimEnd(';') + ';' + ($added -join ';')   # this session too

Write-Host ('Added to the user PATH: ' + ($added -join ', '))
Write-Host ''
Write-Host 'Verify in a NEW terminal with:'
Write-Host '    streamlit --version'
Write-Host '    streamlit run app.py'

