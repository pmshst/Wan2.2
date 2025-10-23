# 🎭 Real-Time Face Swap Extension

## Nueva Extensión Comunitaria: Real-Time Face Swap

Hemos creado una extensión experimental que permite realizar **face swap en tiempo real** usando Wan2.2-Animate con tu webcam.

### 🌟 Características

- ✅ Captura de webcam en tiempo real (30fps)
- ✅ Procesamiento asíncrono con threading
- ✅ Interfaz interactiva con OpenCV
- ✅ Métricas de rendimiento en vivo
- ✅ Controles por teclado
- ✅ Optimizado para diferentes GPUs

### 🚀 Inicio Rápido

```bash
# Ir al directorio de la extensión
cd realtime_face_swap

# Ejecutar ejemplo simple
python example_simple.py

# O usar el launcher
launcher.bat  # Windows
```

### 📚 Documentación Completa

Toda la documentación está en el directorio `realtime_face_swap/`:

- **[README.md](realtime_face_swap/README.md)** - Documentación completa
- **[QUICKSTART.md](realtime_face_swap/QUICKSTART.md)** - Guía de inicio rápido en 5 pasos
- **[ARCHITECTURE.md](realtime_face_swap/ARCHITECTURE.md)** - Arquitectura del sistema
- **[TROUBLESHOOTING.md](realtime_face_swap/TROUBLESHOOTING.md)** - Solución de problemas
- **[PROJECT_SUMMARY.md](realtime_face_swap/PROJECT_SUMMARY.md)** - Resumen del proyecto

### 🎮 Demo

[![Real-Time Face Swap Demo](https://img.shields.io/badge/Demo-Real--Time%20Face%20Swap-blue)](realtime_face_swap/)

**Controles:**
- `ESPACIO`: Pausar/Reanudar
- `O`: Toggle Original/Procesado
- `S`: Toggle Estadísticas
- `Q/ESC`: Salir

### ⚙️ Requisitos

- GPU NVIDIA con 16GB+ VRAM
- Python 3.10+
- Webcam compatible
- Modelos Wan2.2-Animate-14B descargados

### 📊 Rendimiento Esperado

| GPU | Process FPS | Latencia | Configuración |
|-----|-------------|----------|---------------|
| RTX 4090 | 3-5 fps | 2-4s | Batch 8, 1280x720 |
| RTX 3090 | 2-4 fps | 3-5s | Batch 6, 1280x720 |
| RTX 3080 | 1-2 fps | 4-8s | Batch 4, 960x540 |

### 🔧 Configuración Rápida

1. **Instalar dependencias** (si aún no lo hiciste):
   ```bash
   pip install -r requirements_windows.txt
   ```

2. **Descargar modelos**:
   ```bash
   cd C:\models
   git clone https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
   git clone https://huggingface.co/Wan-AI/Wan2.2-Animate-Process
   ```

3. **Editar configuración** en `realtime_face_swap/example_simple.py`:
   ```python
   CHECKPOINT_DIR = r"C:\models\Wan2.2-Animate-14B"
   PREPROCESS_CHECKPOINT_DIR = r"C:\models\Wan2.2-Animate-Process"
   REFERENCE_IMAGE = r"C:\path\to\your\face.jpg"
   ```

4. **Ejecutar**:
   ```bash
   cd realtime_face_swap
   python example_simple.py
   ```

### ⚠️ Nota Importante

Esta es una extensión **experimental** que demuestra las capacidades de Wan2.2-Animate en un escenario de baja latencia. Debido a la complejidad del modelo:

- ❌ NO es verdadero "real-time" a 60fps sin latencia
- ✅ Latencia típica: 2-5 segundos por lote
- ✅ Funciona mejor con GPUs potentes (24GB VRAM)
- ✅ Es un proof-of-concept para aplicaciones interactivas

### 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Esta extensión es parte de la comunidad y puede ser mejorada por cualquiera.

### 📄 Licencia

Sigue la misma licencia que Wan2.2.

---

**📁 Estructura del Proyecto:**
```
Wan2.2/
├── realtime_face_swap/           # 🆕 Extensión de Real-Time Face Swap
│   ├── webcam_capture.py
│   ├── face_swap_processor.py
│   ├── realtime_app.py
│   ├── example_simple.py
│   ├── example_advanced.py
│   ├── test_system.py
│   ├── config.ini
│   ├── launcher.bat
│   └── docs/
│       ├── README.md
│       ├── QUICKSTART.md
│       ├── ARCHITECTURE.md
│       └── TROUBLESHOOTING.md
├── wan/                          # Código principal de Wan2.2
├── examples/                      # Ejemplos oficiales
├── generate.py                    # Script de generación
└── requirements_windows.txt       # Dependencias
```

### 🎯 Casos de Uso

- 🎬 Demos interactivos de tecnología de face swap
- 🔬 Experimentación con IA generativa en vivo
- 🎓 Educación sobre modelos de difusión
- 🛠️ Prototipado rápido de aplicaciones de video
- 🎨 Proyectos creativos y artísticos

### 📞 Soporte

Para problemas específicos de la extensión Real-Time Face Swap:
1. Consulta [TROUBLESHOOTING.md](realtime_face_swap/TROUBLESHOOTING.md)
2. Ejecuta los tests: `python realtime_face_swap/test_system.py`
3. Revisa los [ejemplos](realtime_face_swap/)
4. Abre un Issue en GitHub

---

**¿Interesado en probar?** → [Ir a Real-Time Face Swap](realtime_face_swap/)
