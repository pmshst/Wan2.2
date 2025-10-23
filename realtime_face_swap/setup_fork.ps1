# Script de Configuración de Fork para Real-Time Face Swap
# Este script te ayuda a configurar tu fork personal de Wan2.2

param(
    [Parameter(Mandatory=$true, HelpMessage="Tu nombre de usuario de GitHub")]
    [string]$GitHubUsername
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Configuración de Fork Personal - Wan2.2" -ForegroundColor Cyan
Write-Host " Real-Time Face Swap Extension" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que estamos en el directorio correcto
if (-not (Test-Path "realtime_face_swap")) {
    Write-Host "❌ Error: No se encuentra el directorio 'realtime_face_swap'" -ForegroundColor Red
    Write-Host "   Por favor ejecuta este script desde: C:\dev\Wan2.2" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Directorio verificado" -ForegroundColor Green
Write-Host ""

# Paso 1: Verificar remotes actuales
Write-Host "[1/7] Verificando configuración actual de Git..." -ForegroundColor Yellow
Write-Host ""

git remote -v

Write-Host ""
Write-Host "Configuración actual de remotes mostrada arriba." -ForegroundColor Cyan
Write-Host ""

$confirm = Read-Host "¿Has creado tu fork en GitHub? (S/N)"
if ($confirm -ne "S" -and $confirm -ne "s") {
    Write-Host ""
    Write-Host "⚠ Por favor crea tu fork primero:" -ForegroundColor Yellow
    Write-Host "   1. Ve a: https://github.com/Wan-Video/Wan2.2" -ForegroundColor Cyan
    Write-Host "   2. Haz clic en 'Fork' (esquina superior derecha)" -ForegroundColor Cyan
    Write-Host "   3. Crea el fork con tu cuenta" -ForegroundColor Cyan
    Write-Host "   4. Ejecuta este script nuevamente" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

Write-Host ""
Write-Host "[2/7] Renombrando 'origin' a 'upstream'..." -ForegroundColor Yellow

try {
    git remote rename origin upstream
    Write-Host "✓ Remote 'origin' renombrado a 'upstream'" -ForegroundColor Green
} catch {
    Write-Host "⚠ No se pudo renombrar (puede que ya esté configurado)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[3/7] Agregando tu fork como 'origin'..." -ForegroundColor Yellow

$forkUrl = "https://github.com/$GitHubUsername/Wan2.2.git"
Write-Host "URL del fork: $forkUrl" -ForegroundColor Cyan

try {
    git remote add origin $forkUrl
    Write-Host "✓ Fork agregado como 'origin'" -ForegroundColor Green
} catch {
    Write-Host "⚠ Remote 'origin' ya existe, actualizando URL..." -ForegroundColor Yellow
    git remote set-url origin $forkUrl
    Write-Host "✓ URL de 'origin' actualizada" -ForegroundColor Green
}

Write-Host ""
Write-Host "[4/7] Verificando configuración de remotes..." -ForegroundColor Yellow
Write-Host ""

git remote -v

Write-Host ""
Write-Host "✓ Remotes configurados correctamente" -ForegroundColor Green

Write-Host ""
Write-Host "[5/7] Creando rama 'feature/realtime-face-swap'..." -ForegroundColor Yellow

try {
    $currentBranch = git branch --show-current
    
    if ($currentBranch -eq "feature/realtime-face-swap") {
        Write-Host "✓ Ya estás en la rama 'feature/realtime-face-swap'" -ForegroundColor Green
    } else {
        git checkout -b feature/realtime-face-swap
        Write-Host "✓ Rama 'feature/realtime-face-swap' creada" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ La rama puede ya existir, cambiando a ella..." -ForegroundColor Yellow
    git checkout feature/realtime-face-swap
}

Write-Host ""
Write-Host "[6/7] Verificando archivos a commitear..." -ForegroundColor Yellow
Write-Host ""

git status

Write-Host ""
$confirmAdd = Read-Host "¿Agregar archivos de 'realtime_face_swap/' al stage? (S/N)"

if ($confirmAdd -eq "S" -or $confirmAdd -eq "s") {
    Write-Host ""
    Write-Host "Agregando archivos..." -ForegroundColor Cyan
    
    git add realtime_face_swap/
    
    Write-Host "✓ Archivos agregados" -ForegroundColor Green
    Write-Host ""
    Write-Host "Estado actual:" -ForegroundColor Cyan
    git status --short
}

Write-Host ""
Write-Host "[7/7] ¿Hacer commit y push?" -ForegroundColor Yellow
Write-Host ""

$confirmCommit = Read-Host "¿Crear commit y hacer push a tu fork? (S/N)"

if ($confirmCommit -eq "S" -or $confirmCommit -eq "s") {
    Write-Host ""
    Write-Host "Creando commit..." -ForegroundColor Cyan
    
    $commitMessage = @"
Add Real-Time Face Swap extension

- Implement webcam capture with circular buffer and threading
- Add face swap processor with Wan2.2-Animate integration
- Create real-time application with asynchronous processing
- Add comprehensive documentation and guides
- Include configuration files and examples
- Add unit tests and verification scripts

Features:
- 30fps webcam capture
- Asynchronous face swap processing with batch optimization
- Interactive UI with OpenCV and keyboard controls
- Performance metrics (FPS, latency)
- GPU optimization profiles for different hardware
- Support for RTX 4090, 3090, 3080, 3070, 3060

Documentation:
- README.md with complete installation and usage guide
- QUICKSTART.md with 5-step quick start
- ARCHITECTURE.md with system architecture diagrams
- TROUBLESHOOTING.md with problem-solving guide
- CHANGELOG.md with version history

Components:
- webcam_capture.py: Real-time webcam capture
- face_swap_processor.py: Face swap processing with Wan2.2
- realtime_app.py: Main application with UI
- example_simple.py and example_advanced.py: Usage examples
- test_system.py: Unit tests
- setup_and_verify.ps1: Setup script
"@

    git commit -m $commitMessage
    
    Write-Host "✓ Commit creado" -ForegroundColor Green
    Write-Host ""
    Write-Host "Haciendo push a tu fork..." -ForegroundColor Cyan
    
    try {
        git push -u origin feature/realtime-face-swap
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host " ✓ ¡Push completado exitosamente!" -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Tu extensión Real-Time Face Swap está ahora en tu fork:" -ForegroundColor Cyan
        Write-Host "  https://github.com/$GitHubUsername/Wan2.2" -ForegroundColor White
        Write-Host ""
        Write-Host "Rama: feature/realtime-face-swap" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Próximos pasos:" -ForegroundColor Yellow
        Write-Host "  1. Ve a tu fork en GitHub" -ForegroundColor White
        Write-Host "  2. Verás un botón 'Compare & pull request'" -ForegroundColor White
        Write-Host "  3. Puedes crear un PR si quieres proponer tus cambios" -ForegroundColor White
        Write-Host "  4. O simplemente usar tu fork de forma independiente" -ForegroundColor White
        Write-Host ""
    } catch {
        Write-Host "❌ Error al hacer push:" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host ""
        Write-Host "Posibles soluciones:" -ForegroundColor Yellow
        Write-Host "  - Verifica que el fork existe en GitHub" -ForegroundColor White
        Write-Host "  - Verifica tus credenciales de Git" -ForegroundColor White
        Write-Host "  - Intenta: git push -u origin feature/realtime-face-swap" -ForegroundColor White
    }
} else {
    Write-Host ""
    Write-Host "⚠ Commit y push cancelados" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Puedes hacerlo manualmente más tarde con:" -ForegroundColor Cyan
    Write-Host "  git commit -m 'Add Real-Time Face Swap extension'" -ForegroundColor White
    Write-Host "  git push -u origin feature/realtime-face-swap" -ForegroundColor White
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Configuración Completada" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Resumen de configuración:" -ForegroundColor Yellow
Write-Host ""
git remote -v
Write-Host ""
git branch
Write-Host ""

Write-Host "Para más información, consulta:" -ForegroundColor Cyan
Write-Host "  - realtime_face_swap/GIT_FORK_GUIDE.md" -ForegroundColor White
Write-Host ""
