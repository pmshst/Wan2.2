# Script de Instalación y Verificación
# Real-Time Face Swap para Wan2.2

Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Real-Time Face Swap - Setup & Verification" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar Python
Write-Host "[1/6] Verificando Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Python encontrado: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python no encontrado. Por favor instala Python 3.10+" -ForegroundColor Red
    exit 1
}

# Verificar CUDA
Write-Host ""
Write-Host "[2/6] Verificando CUDA..." -ForegroundColor Yellow
try {
    $nvccVersion = nvcc --version 2>&1 | Select-String "release"
    Write-Host "  ✓ CUDA encontrado: $nvccVersion" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ CUDA no encontrado. El sistema requerirá GPU con CUDA." -ForegroundColor Yellow
}

# Verificar GPU
Write-Host ""
Write-Host "[3/6] Verificando GPU NVIDIA..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ GPU encontrada:" -ForegroundColor Green
        Write-Host "    $gpuInfo" -ForegroundColor Cyan
    } else {
        Write-Host "  ✗ No se detectó GPU NVIDIA" -ForegroundColor Red
    }
} catch {
    Write-Host "  ✗ nvidia-smi no disponible" -ForegroundColor Red
}

# Verificar dependencias de Python
Write-Host ""
Write-Host "[4/6] Verificando dependencias de Python..." -ForegroundColor Yellow

$requiredPackages = @(
    "torch",
    "torchvision",
    "opencv-python",
    "numpy",
    "decord",
    "diffusers",
    "transformers",
    "accelerate",
    "einops",
    "peft"
)

$missingPackages = @()

foreach ($package in $requiredPackages) {
    try {
        python -c "import $($package.Replace('-', '_'))" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ $package instalado" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $package no encontrado" -ForegroundColor Red
            $missingPackages += $package
        }
    } catch {
        Write-Host "  ✗ $package no encontrado" -ForegroundColor Red
        $missingPackages += $package
    }
}

# Instalar dependencias faltantes
if ($missingPackages.Count -gt 0) {
    Write-Host ""
    Write-Host "[5/6] Instalando dependencias faltantes..." -ForegroundColor Yellow
    
    $install = Read-Host "¿Deseas instalar las dependencias faltantes? (S/N)"
    
    if ($install -eq "S" -or $install -eq "s") {
        Write-Host "  Instalando requirements_windows.txt..." -ForegroundColor Cyan
        
        $parentDir = Split-Path -Parent $PSScriptRoot
        $requirementsPath = Join-Path $parentDir "requirements_windows.txt"
        
        if (Test-Path $requirementsPath) {
            python -m pip install -r $requirementsPath
            Write-Host "  ✓ Dependencias instaladas" -ForegroundColor Green
        } else {
            Write-Host "  ✗ No se encontró requirements_windows.txt" -ForegroundColor Red
            Write-Host "    Por favor ejecuta: pip install -r ../requirements_windows.txt" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  ✓ Todas las dependencias están instaladas" -ForegroundColor Green
}

# Verificar estructura de directorios
Write-Host ""
Write-Host "[6/6] Verificando estructura del proyecto..." -ForegroundColor Yellow

$requiredFiles = @(
    "webcam_capture.py",
    "face_swap_processor.py",
    "realtime_app.py",
    "example_simple.py",
    "__init__.py",
    "README.md"
)

$allFilesExist = $true

foreach ($file in $requiredFiles) {
    $filePath = Join-Path $PSScriptRoot $file
    if (Test-Path $filePath) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file no encontrado" -ForegroundColor Red
        $allFilesExist = $false
    }
}

# Resumen
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Resumen de Verificación" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

if ($allFilesExist -and $missingPackages.Count -eq 0) {
    Write-Host "✓ Sistema listo para usar!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Siguiente paso:" -ForegroundColor Yellow
    Write-Host "  1. Descarga los modelos (ver QUICKSTART.md)" -ForegroundColor Cyan
    Write-Host "  2. Edita example_simple.py con las rutas correctas" -ForegroundColor Cyan
    Write-Host "  3. Ejecuta: python example_simple.py" -ForegroundColor Cyan
} else {
    Write-Host "⚠ Hay problemas que resolver antes de continuar" -ForegroundColor Yellow
    
    if ($missingPackages.Count -gt 0) {
        Write-Host ""
        Write-Host "Paquetes faltantes:" -ForegroundColor Red
        foreach ($pkg in $missingPackages) {
            Write-Host "  - $pkg" -ForegroundColor Red
        }
    }
    
    if (-not $allFilesExist) {
        Write-Host ""
        Write-Host "Archivos faltantes del proyecto" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Test de importación opcional
Write-Host "¿Deseas ejecutar un test de importación? (S/N): " -NoNewline -ForegroundColor Yellow
$runTest = Read-Host

if ($runTest -eq "S" -or $runTest -eq "s") {
    Write-Host ""
    Write-Host "Ejecutando test de importación..." -ForegroundColor Cyan
    
    python -c @"
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

try:
    print('Importando módulos...')
    from realtime_face_swap import WebcamCapture, FramePreprocessor
    from realtime_face_swap import RealtimeFaceSwap, AsyncFaceSwapProcessor
    from realtime_face_swap import RealtimeFaceSwapApp
    print('✓ Todos los módulos importados correctamente')
except Exception as e:
    print(f'✗ Error al importar módulos: {e}')
    sys.exit(1)
"@

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Test de importación exitoso" -ForegroundColor Green
    } else {
        Write-Host "✗ Test de importación falló" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Para más información, consulta:" -ForegroundColor Cyan
Write-Host "  - README.md (documentación completa)" -ForegroundColor Cyan
Write-Host "  - QUICKSTART.md (guía de inicio rápido)" -ForegroundColor Cyan
Write-Host ""
