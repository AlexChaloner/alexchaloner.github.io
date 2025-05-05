# 1) Ensure all installed
if (-not (Get-Command htmlproofer -ErrorAction SilentlyContinue)) {
    Write-Host "Installing html-proofer..." -ForegroundColor Cyan
    gem install html-proofer
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Failed to install html-proofer"
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
Write-Host "Running link checker..." -ForegroundColor Cyan
htmlproofer .\_site --allow-hash-href --check-external --check-img-http
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Broken links detected"
    exit 1
}

Write-Host "✅ All links OK!" -ForegroundColor Green