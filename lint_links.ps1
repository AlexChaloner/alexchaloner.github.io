# 1) Ensure all installed
if (-not (Get-Command linkinator -ErrorAction SilentlyContinue)) {
    Write-Host "Installing linkinator..." -ForegroundColor Cyan
    npm install linkinator
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Failed to install linkinator"
        exit 1
    }
}

if (-not (Get-Command jekyll -ErrorAction SilentlyContinue)) {
    Write-Host "Installing jekyll..." -ForegroundColor Cyan
    gem install jekyll
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Failed to install jekyll"
        exit 1
    }
}

# 2) Build your Jekyll site
Write-Host "Building Jekyll site..." -ForegroundColor Cyan
jekyll build
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Jekyll build failed"
    exit 1
}

# 3) Run html-proofer on the generated folder
Write-Host "Running linkinator..." -ForegroundColor Cyan
linkinator .\_site --concurrency 20
if ($LASTEXITCODE -ne 0) {
    exit 1
}

Write-Host "✅ All links OK!" -ForegroundColor Green
exit 0
