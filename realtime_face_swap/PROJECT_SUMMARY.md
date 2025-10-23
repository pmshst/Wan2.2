# 🎭 Real-Time Face Swap con Wan2.2

## 📁 Estructura del Proyecto

```
realtime_face_swap/
├── 📄 __init__.py                    # Inicialización del paquete
├── 📄 webcam_capture.py              # Captura de webcam en tiempo real
├── 📄 face_swap_processor.py         # Procesador de face swap
├── 📄 realtime_app.py                # Aplicación principal
├── 📄 example_simple.py              # Ejemplo simple de uso
├── 📄 example_advanced.py            # Ejemplo avanzado con config
├── 📄 test_system.py                 # Tests unitarios
├── 📄 setup_and_verify.ps1           # Script de setup (PowerShell)
├── 📄 config.ini                     # Archivo de configuración
├── 📘 README.md                      # Documentación principal
├── 📗 QUICKSTART.md                  # Guía de inicio rápido
├── 📙 ARCHITECTURE.md                # Documentación de arquitectura
├── 📕 TROUBLESHOOTING.md             # Solución de problemas
└── 📔 CHANGELOG.md                   # Historial de cambios
```

## 🚀 Inicio Rápido (3 pasos)

### 1. Descargar Modelos
```powershell
cd C:\models
git clone https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
git clone https://huggingface.co/Wan-AI/Wan2.2-Animate-Process
```

### 2. Configurar
Edita `example_simple.py`:
```python
CHECKPOINT_DIR = r"C:\models\Wan2.2-Animate-14B"
PREPROCESS_CHECKPOINT_DIR = r"C:\models\Wan2.2-Animate-Process"
REFERENCE_IMAGE = r"C:\path\to\your\face.jpg"
```

### 3. Ejecutar
```powershell
cd C:\dev\Wan2.2\realtime_face_swap
python example_simple.py
```

## 📚 Documentación

| Documento | Descripción |
|-----------|-------------|
| [README.md](README.md) | Documentación completa con instalación, uso y ejemplos |
| [QUICKSTART.md](QUICKSTART.md) | Guía rápida de 5 pasos para empezar |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Arquitectura del sistema, flujos de datos y componentes |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Solución de problemas comunes y optimización |
| [CHANGELOG.md](CHANGELOG.md) | Historial de versiones y cambios |

## 🎯 Características Principales

✅ **Captura en Tiempo Real**
- Captura de webcam a 30fps
- Buffer circular para gestión eficiente
- Control de resolución y FPS

✅ **Procesamiento Asíncrono**
- Threading para operaciones no bloqueantes
- Procesamiento por lotes
- Latencia de 2-5 segundos

✅ **Interfaz Interactiva**
- Visualización en tiempo real
- Controles por teclado
- Estadísticas de rendimiento

✅ **Optimizado**
- Configuraciones por tipo de GPU
- Caché de referencia
- Gestión inteligente de memoria

## 🎮 Controles

| Tecla | Acción |
|-------|--------|
| `ESPACIO` | Pausar/Reanudar |
| `O` | Ver Original/Procesado |
| `S` | Mostrar/Ocultar Stats |
| `Q` o `ESC` | Salir |

## ⚙️ Configuración Según GPU

### RTX 4090 (24GB)
```python
BATCH_SIZE = 8
WIDTH = 1280
HEIGHT = 720
```

### RTX 3090 (24GB)
```python
BATCH_SIZE = 6
WIDTH = 1280
HEIGHT = 720
```

### RTX 3080/3070 (10-12GB)
```python
BATCH_SIZE = 4
WIDTH = 960
HEIGHT = 540
```

## 🧪 Verificación y Tests

```powershell
# Verificar instalación
.\setup_and_verify.ps1

# Ejecutar tests
python test_system.py
```

## 📊 Métricas Esperadas

**RTX 4090:**
- Capture FPS: 30
- Process FPS: 3-5
- Latency: 2-4s/lote

## 🔧 Ejemplos de Uso

### Ejemplo Simple
```powershell
python example_simple.py
```

### Ejemplo Avanzado con Config
```powershell
python example_advanced.py --config config.ini
```

### Con Parámetros Personalizados
```powershell
python example_advanced.py --batch_size 4 --width 960 --height 540
```

## 🐛 Problemas Comunes

### CUDA Out of Memory
```python
# Reduce batch_size y resolución
BATCH_SIZE = 2
WIDTH = 640
HEIGHT = 480
```

### Cámara No Encontrada
```python
# Prueba diferentes IDs
CAMERA_ID = 0  # o 1, 2, 3...
```

Ver [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para más soluciones.

## 📦 Componentes del Sistema

### 1. WebcamCapture
Captura frames de webcam sin bloqueo usando threading.

### 2. RealtimeFaceSwap  
Procesador principal de face swap con Wan2.2-Animate.

### 3. AsyncFaceSwapProcessor
Gestión asíncrona de procesamiento en thread separado.

### 4. RealtimeFaceSwapApp
Aplicación principal con coordinación y UI.

## 🛠️ Tecnologías

- Python 3.10+
- PyTorch 2.4+
- OpenCV
- Wan2.2-Animate-14B
- CUDA 11.8+

## 📋 Requisitos

- GPU NVIDIA 16GB+ VRAM
- RAM 32GB+ recomendado  
- Webcam compatible
- Windows 10/11

## 🔄 Workflow

```
Webcam (30fps)
    ↓
Buffer Circular
    ↓
Queue de Procesamiento
    ↓
Procesamiento por Lotes (Wan2.2)
    ↓
Buffer de Salida
    ↓
Display (30fps)
```

## 🎯 Casos de Uso

- Demos en vivo de IA
- Experimentación con face swap
- Prototipado de aplicaciones
- Investigación en video processing
- Educación sobre modelos de difusión

## ⚠️ Limitaciones

- NO es verdadero real-time 60fps
- Latencia de 2-5 segundos inevitable
- Requiere GPU potente
- Trade-off calidad vs velocidad

## 📞 Soporte

1. Lee [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Ejecuta `python test_system.py`
3. Busca en GitHub Issues
4. Crea un nuevo Issue

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas!

1. Fork del repositorio
2. Crea tu feature branch
3. Commit tus cambios
4. Push a la branch
5. Abre un Pull Request

## 📄 Licencia

Extiende [Wan2.2](https://github.com/Wan-Video/Wan2.2) con la misma licencia.

## 🙏 Agradecimientos

- Equipo Alibaba Wan por Wan2.2-Animate
- Comunidad open source
- Todos los contribuidores

## 📈 Roadmap

### v1.1.0 (Próximo)
- [ ] Optimización de preprocesamiento
- [ ] Soporte FP16/BF16
- [ ] Modo de grabación
- [ ] Múltiples referencias

### v2.0.0 (Futuro)
- [ ] Soporte multi-GPU
- [ ] Face swap múltiple
- [ ] Plugin OBS Studio
- [ ] API REST

## 🌟 Star el Proyecto

Si te resulta útil, ¡dale una estrella en GitHub! ⭐

---

**Versión:** 1.0.0  
**Fecha:** 23 de octubre de 2025  
**Autor:** Real-time Face Swap Extension Team

**¿Listo para empezar?** → Lee [QUICKSTART.md](QUICKSTART.md)
