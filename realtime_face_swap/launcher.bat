@echo off
REM Real-Time Face Swap - Script de Inicio Rápido
REM Copyright 2024-2025

setlocal EnableDelayedExpansion

echo ================================================================
echo   REAL-TIME FACE SWAP - LAUNCHER
echo   Powered by Wan2.2-Animate
echo ================================================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado
    echo Por favor instala Python 3.10 o superior
    pause
    exit /b 1
)

REM Menu principal
:MENU
echo.
echo Selecciona una opcion:
echo.
echo   1. Ejecutar ejemplo simple
echo   2. Ejecutar ejemplo avanzado
echo   3. Ejecutar tests
echo   4. Verificar sistema
echo   5. Ver ayuda
echo   6. Salir
echo.
set /p choice="Opcion (1-6): "

if "%choice%"=="1" goto RUN_SIMPLE
if "%choice%"=="2" goto RUN_ADVANCED
if "%choice%"=="3" goto RUN_TESTS
if "%choice%"=="4" goto RUN_VERIFY
if "%choice%"=="5" goto SHOW_HELP
if "%choice%"=="6" goto EXIT

echo Opcion invalida
goto MENU

:RUN_SIMPLE
echo.
echo ================================================================
echo   Ejecutando ejemplo simple...
echo ================================================================
echo.
python example_simple.py
goto MENU

:RUN_ADVANCED
echo.
echo ================================================================
echo   Ejecutando ejemplo avanzado...
echo ================================================================
echo.
python example_advanced.py --config config.ini
goto MENU

:RUN_TESTS
echo.
echo ================================================================
echo   Ejecutando tests del sistema...
echo ================================================================
echo.
python test_system.py
pause
goto MENU

:RUN_VERIFY
echo.
echo ================================================================
echo   Verificando sistema...
echo ================================================================
echo.
powershell -ExecutionPolicy Bypass -File setup_and_verify.ps1
pause
goto MENU

:SHOW_HELP
echo.
echo ================================================================
echo   AYUDA - Real-Time Face Swap
echo ================================================================
echo.
echo ARCHIVOS PRINCIPALES:
echo   - example_simple.py: Script simple para empezar rapido
echo   - example_advanced.py: Script con opciones avanzadas
echo   - config.ini: Archivo de configuracion
echo.
echo DOCUMENTACION:
echo   - README.md: Documentacion completa
echo   - QUICKSTART.md: Guia rapida
echo   - TROUBLESHOOTING.md: Solucion de problemas
echo   - ARCHITECTURE.md: Arquitectura del sistema
echo.
echo COMANDOS MANUALES:
echo   python example_simple.py
echo   python example_advanced.py --config config.ini
echo   python test_system.py
echo.
echo CONTROLES EN LA APP:
echo   - ESPACIO: Pausar/Reanudar
echo   - O: Toggle Original/Procesado
echo   - S: Toggle Estadisticas
echo   - Q/ESC: Salir
echo.
pause
goto MENU

:EXIT
echo.
echo Hasta luego!
exit /b 0
