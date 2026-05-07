# Production Sandbox Build Script (Windows PowerShell)
# Usage: .\build_sandbox.ps1 [-ImageName "castalia-sandbox:latest"]
#
# Builds the hardened Docker sandbox image with all security layers.

param(
    [string]$ImageName = "castalia-sandbox:latest",
    [string]$Dockerfile = "sandbox.Dockerfile",
    [switch]$SkipTests = $false,
    [switch]$Push = $false,
    [string]$Registry = ""
)

$ErrorActionPreference = "Stop"

function Write-Header($text) {
    Write-Host "`n=== $text ===" -ForegroundColor Cyan
}

function Write-Success($text) {
    Write-Host "  [PASS] $text" -ForegroundColor Green
}

function Write-Fail($text) {
    Write-Host "  [FAIL] $text" -ForegroundColor Red
}

Write-Header "Castalia Production Sandbox Build"

# ── Check Prerequisites ──────────────────────────────────────────
Write-Header "Checking Prerequisites"

try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Docker not found" }
    Write-Success "Docker installed: $dockerVersion"
} catch {
    Write-Fail "Docker not found. Install Docker Desktop first."
    exit 1
}

try {
    $dockerInfo = docker info --format "{{.ServerVersion}}" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Docker daemon not running" }
    Write-Success "Docker daemon running: $dockerInfo"
} catch {
    Write-Fail "Docker daemon not running. Start Docker Desktop."
    exit 1
}

# ── Validate Files ─────────────────────────────────────────────
Write-Header "Validating Build Files"

$requiredFiles = @($Dockerfile, "seccomp-profile.json")
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length
        Write-Success "$file exists ($size bytes)"
    } else {
        Write-Fail "$file not found!"
        exit 1
    }
}

# ── Build Image ────────────────────────────────────────────────
Write-Header "Building Sandbox Image: $ImageName"

try {
    docker build -f $Dockerfile -t $ImageName . --no-cache
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
    Write-Success "Image built successfully"
} catch {
    Write-Fail "Docker build failed: $_"
    exit 1
}

# ── Verify Image ───────────────────────────────────────────────
Write-Header "Verifying Image"

try {
    $imageInfo = docker images $ImageName --format "{{.Size}}"
    if ($imageInfo) {
        Write-Success "Image size: $imageInfo"
    } else {
        Write-Fail "Image not found after build"
        exit 1
    }
} catch {
    Write-Fail "Image verification failed: $_"
    exit 1
}

# ── Security Audit (Image Contents) ────────────────────────────
Write-Header "Security Audit: Image Contents"

try {
    # Check for dangerous binaries
    $dangerousBins = @("curl", "wget", "ssh", "nc", "ncat", "telnet", "python3-pip")
    foreach ($bin in $dangerousBins) {
        $result = docker run --rm $ImageName which $bin 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Fail "DANGEROUS BINARY FOUND: $bin"
        } else {
            Write-Success "$bin not present (good)"
        }
    }

    # Check user is non-root
    $user = docker run --rm $ImageName id -u 2>$null
    if ($user.Trim() -eq "0") {
        Write-Fail "Container runs as root (UID 0)!"
    } else {
        Write-Success "Container runs as non-root user: UID $user"
    }

    # Check pip is removed
    $pipCheck = docker run --rm $ImageName pip --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Fail "pip is still installed!"
    } else {
        Write-Success "pip removed (good)"
    }

    # Check Python works
    $pyCheck = docker run --rm $ImageName python -c "print('OK')" 2>$null
    if ($pyCheck.Trim() -eq "OK") {
        Write-Success "Python interpreter functional"
    } else {
        Write-Fail "Python interpreter not functional!"
        exit 1
    }
} catch {
    Write-Fail "Security audit failed: $_"
    exit 1
}

# ── Functional Tests ─────────────────────────────────────────────
if (-not $SkipTests) {
    Write-Header "Running Functional Tests"

    # Test 1: Basic execution
    Write-Host "Test 1: Basic execution..." -NoNewline
    $result = docker run --rm --network none --memory 128m --read-only `
        --cap-drop ALL --security-opt no-new-privileges:true `
        -v "${PWD}\test_basic.py:/app/code.py:ro" $ImageName 2>&1
    if ($LASTEXITCODE -eq 0 -and $result -match "Hello from sandbox") {
        Write-Success "Basic execution"
    } else {
        Write-Fail "Basic execution failed: $result"
        exit 1
    }

    # Test 2: Network isolation
    Write-Host "Test 2: Network isolation..." -NoNewline
    $result = docker run --rm --network none --memory 128m --read-only `
        -v "${PWD}\test_network.py:/app/code.py:ro" $ImageName 2>&1
    if ($LASTEXITCODE -ne 0 -or $result -match "error" -or $result -match "Error") {
        Write-Success "Network blocked (expected failure)"
    } else {
        Write-Fail "Network NOT blocked!"
        exit 1
    }

    # Test 3: Filesystem read-only
    Write-Host "Test 3: Filesystem read-only..." -NoNewline
    $result = docker run --rm --network none --memory 128m --read-only `
        -v "${PWD}\test_write.py:/app/code.py:ro" $ImageName 2>&1
    if ($LASTEXITCODE -ne 0 -or $result -match "Read-only" -or $result -match "Permission") {
        Write-Success "Write blocked (expected failure)"
    } else {
        Write-Fail "Write NOT blocked!"
        exit 1
    }
}

# ── Push (optional) ────────────────────────────────────────────
if ($Push -and $Registry) {
    Write-Header "Pushing to Registry"
    $fullName = "$Registry/$ImageName"
    docker tag $ImageName $fullName
    docker push $fullName
    Write-Success "Pushed to $fullName"
}

# ── Summary ────────────────────────────────────────────────────
Write-Header "Build Complete"
Write-Host "Image: $ImageName" -ForegroundColor White
Write-Host "Dockerfile: $Dockerfile" -ForegroundColor White
Write-Host "Seccomp: seccomp-profile.json" -ForegroundColor White
Write-Host "`nTo use in Python:" -ForegroundColor Yellow
Write-Host "  sandbox = DockerSandbox(image='$ImageName')" -ForegroundColor White
Write-Host "  result = sandbox.execute('print(2+2)')" -ForegroundColor White
