# Real-Time Face Swap con Wan2.2

Sistema de intercambio de rostros en tiempo real utilizando la tecnología Wan2.2-Animate.

## 🌟 Características

- ✅ **Captura en tiempo real**: Captura video desde webcam con control de FPS y buffer circular
- ✅ **Procesamiento asíncrono**: Procesamiento de face swap en thread separado para minimizar latencia
- ✅ **Procesamiento por lotes**: Optimización mediante procesamiento de múltiples frames simultáneamente
- ✅ **Interfaz interactiva**: Visualización en tiempo real con controles y estadísticas
- ✅ **Métricas de rendimiento**: Monitoreo de FPS, latencia y uso de recursos

## 📋 Requisitos

### Hardware
- **GPU**: NVIDIA con al menos 16GB VRAM (recomendado: RTX 3090, RTX 4090, A100)
- **RAM**: 32GB+ recomendado
- **Webcam**: Cualquier cámara compatible con OpenCV

### Software
- Python 3.10+
- CUDA 11.8+ y cuDNN
- Todas las dependencias de Wan2.2 (ver requirements.txt principal)

## 🚀 Instalación

1. **Instalar dependencias de Wan2.2** (si no lo has hecho):
```powershell
cd C:\dev\Wan2.2
pip install -r requirements_windows.txt
```

2. **Descargar checkpoints del modelo**:

Necesitas descargar los siguientes modelos:
- **Wan2.2-Animate-14B**: Modelo principal de animación
- **Checkpoints de preprocesamiento**: Detección de pose, face detection y SAM2

Descarga desde:
- [Hugging Face - Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- [Hugging Face - Wan-AI/Wan2.2-Animate-Process](https://huggingface.co/Wan-AI/Wan2.2-Animate-Process)

O usando git-lfs:
```powershell
# Modelo principal
git clone https://huggingface.co/Wan-AI/Wan2.2-Animate-14B

# Checkpoints de preprocesamiento
git clone https://huggingface.co/Wan-AI/Wan2.2-Animate-Process
```

## 📖 Uso

### Modo Básico

```powershell
cd C:\dev\Wan2.2\realtime_face_swap

python realtime_app.py `
    --checkpoint_dir "C:\path\to\Wan2.2-Animate-14B" `
    --preprocess_checkpoint_dir "C:\path\to\Wan2.2-Animate-Process" `
    --reference_image "C:\path\to\reference_face.jpg" `
    --config "../wan/configs/wan_animate_14B.py"
```

### Opciones Avanzadas

```powershell
python realtime_app.py `
    --checkpoint_dir "C:\path\to\Wan2.2-Animate-14B" `
    --preprocess_checkpoint_dir "C:\path\to\Wan2.2-Animate-Process" `
    --reference_image "C:\path\to\reference_face.jpg" `
    --config "../wan/configs/wan_animate_14B.py" `
    --camera_id 0 `
    --width 1280 `
    --height 720 `
    --fps 30 `
    --batch_size 8 `
    --device_id 0
```

### Parámetros

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--checkpoint_dir` | Directorio con el modelo Wan2.2-Animate-14B | Requerido |
| `--preprocess_checkpoint_dir` | Directorio con checkpoints de preprocesamiento | Requerido |
| `--reference_image` | Imagen de referencia para face swap | Requerido |
| `--config` | Archivo de configuración del modelo | Requerido |
| `--camera_id` | ID de la cámara | 0 |
| `--width` | Ancho de captura | 1280 |
| `--height` | Alto de captura | 720 |
| `--fps` | FPS objetivo | 30 |
| `--batch_size` | Tamaño del lote para procesamiento | 8 |
| `--device_id` | ID del dispositivo GPU | 0 |

## 🎮 Controles

Durante la ejecución, puedes usar las siguientes teclas:

| Tecla | Función |
|-------|---------|
| `ESPACIO` | Pausar/Reanudar procesamiento |
| `O` | Alternar entre vista original y procesada |
| `S` | Mostrar/Ocultar estadísticas |
| `Q` o `ESC` | Salir de la aplicación |

## 🏗️ Arquitectura

El sistema consta de tres componentes principales:

### 1. WebcamCapture (`webcam_capture.py`)
- Captura frames desde la webcam en un thread dedicado
- Gestiona un buffer circular de frames
- Controla FPS y resolución
- Proporciona estadísticas de captura

### 2. RealtimeFaceSwap (`face_swap_processor.py`)
- Preprocesa frames (detección de pose, face detection, máscaras)
- Aplica el modelo Wan2.2-Animate para face swap
- Procesa frames en lotes para eficiencia
- Cachea datos de la imagen de referencia

### 3. RealtimeFaceSwapApp (`realtime_app.py`)
- Coordina captura y procesamiento
- Gestiona threads para I/O asíncrono
- Renderiza interfaz de usuario
- Maneja controles y estadísticas

## ⚡ Optimización de Rendimiento

### Configuración Recomendada

Para **RTX 4090 (24GB VRAM)**:
```powershell
--batch_size 8 --clip_len 25 --width 1280 --height 720
```

Para **RTX 3090 (24GB VRAM)**:
```powershell
--batch_size 6 --clip_len 21 --width 1280 --height 720
```

Para **GPU con menos VRAM (16GB)**:
```powershell
--batch_size 4 --clip_len 17 --width 960 --height 540
```

### Tips de Optimización

1. **Reducir batch_size**: Menor latencia pero menor throughput
2. **Reducir clip_len**: Debe ser 4n+1 (ej: 13, 17, 21, 25)
3. **Reducir resolución**: Más rápido pero menor calidad
4. **Usar offload_model**: Para GPUs con poca VRAM (más lento)

## 🔧 Solución de Problemas

### Error: "CUDA out of memory"
- Reduce `batch_size` (ej: de 8 a 4)
- Reduce `clip_len` (ej: de 25 a 17)
- Reduce resolución (`--width 960 --height 540`)
- Cierra otras aplicaciones que usen GPU

### Latencia Alta
- Reduce `batch_size` para procesamiento más frecuente
- Usa una GPU más potente
- Reduce `clip_len` para clips más cortos
- Verifica que no haya otros procesos usando la GPU

### Webcam No Detectada
- Verifica `--camera_id` (prueba con 0, 1, 2...)
- Asegúrate de que ninguna otra app use la cámara
- Verifica permisos de cámara en Windows

### Baja Calidad de Face Swap
- Usa una imagen de referencia de alta calidad
- Asegúrate de que la iluminación sea buena
- La cara de referencia debe estar bien centrada y clara
- Aumenta `sampling_steps` en `face_swap_processor.py` (línea 180)

## 📊 Métricas y Estadísticas

La interfaz muestra las siguientes métricas en tiempo real:

- **Display FPS**: Frames por segundo de visualización
- **Capture FPS**: Frames por segundo de captura de webcam
- **Process FPS**: Frames procesados por segundo
- **Latency**: Tiempo de procesamiento por lote

## 🎯 Ejemplos de Uso

### Ejemplo 1: Demo Rápido
Prueba rápida con configuración por defecto:
```powershell
python realtime_app.py `
    --checkpoint_dir "C:\models\Wan2.2-Animate-14B" `
    --preprocess_checkpoint_dir "C:\models\Wan2.2-Animate-Process" `
    --reference_image "..\examples\wan_animate\replace\image.jpeg" `
    --config "../wan/configs/wan_animate_14B.py"
```

### Ejemplo 2: Alta Calidad
Máxima calidad con GPU potente:
```powershell
python realtime_app.py `
    --checkpoint_dir "C:\models\Wan2.2-Animate-14B" `
    --preprocess_checkpoint_dir "C:\models\Wan2.2-Animate-Process" `
    --reference_image "my_reference.jpg" `
    --config "../wan/configs/wan_animate_14B.py" `
    --width 1920 --height 1080 `
    --batch_size 12
```

### Ejemplo 3: Baja Latencia
Configuración optimizada para latencia mínima:
```powershell
python realtime_app.py `
    --checkpoint_dir "C:\models\Wan2.2-Animate-14B" `
    --preprocess_checkpoint_dir "C:\models\Wan2.2-Animate-Process" `
    --reference_image "my_reference.jpg" `
    --config "../wan/configs/wan_animate_14B.py" `
    --batch_size 2 `
    --width 640 --height 480
```

## 📝 Notas Técnicas

### Limitaciones Conocidas

1. **Latencia**: El modelo Wan2.2 es pesado y no puede lograr verdadero "real-time" (60fps). La latencia típica es de 2-5 segundos por lote.

2. **Calidad vs Velocidad**: Hay un trade-off entre calidad del face swap y velocidad de procesamiento.

3. **Requisitos de Hardware**: Se necesita una GPU potente para rendimiento aceptable.

### Diferencias con el Modo Estándar

Este sistema de real-time face swap:
- ✅ Procesa video continuo de webcam
- ✅ Usa procesamiento asíncrono y por lotes
- ✅ Proporciona feedback visual inmediato
- ❌ Mayor latencia que el procesamiento offline
- ❌ Calidad ligeramente inferior por optimizaciones

## 🤝 Contribuciones

Este es un proyecto experimental que extiende Wan2.2 para uso en tiempo real. Las contribuciones son bienvenidas:

1. Fork del repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto extiende [Wan2.2](https://github.com/Wan-Video/Wan2.2) y sigue la misma licencia.

## 🙏 Agradecimientos

- Equipo de Alibaba Wan por crear Wan2.2
- Comunidad de código abierto por las herramientas utilizadas

## 📧 Soporte

Para problemas o preguntas:
1. Revisa la sección de Solución de Problemas
2. Busca en los Issues de GitHub
3. Crea un nuevo Issue con detalles completos

---

**¡Disfruta tu sistema de Real-Time Face Swap con Wan2.2!** 🎉
