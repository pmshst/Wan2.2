# Guía de Inicio Rápido - Real-Time Face Swap

## 🚀 Inicio Rápido en 5 Pasos

### Paso 1: Descargar Modelos

Descarga los modelos necesarios:

**Opción A: Usando git-lfs (Recomendado)**
```powershell
# Modelo principal Wan2.2-Animate-14B
cd C:\models
git clone https://huggingface.co/Wan-AI/Wan2.2-Animate-14B

# Checkpoints de preprocesamiento
git clone https://huggingface.co/Wan-AI/Wan2.2-Animate-Process
```

**Opción B: Descarga Manual**
- Visita [Hugging Face - Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- Visita [Hugging Face - Wan2.2-Animate-Process](https://huggingface.co/Wan-AI/Wan2.2-Animate-Process)
- Descarga todos los archivos en directorios separados

### Paso 2: Preparar Imagen de Referencia

Necesitas una foto de la cara que quieres usar para el swap:

**Requisitos:**
- ✅ Cara frontal y centrada
- ✅ Buena iluminación
- ✅ Alta resolución (al menos 512x512)
- ✅ Expresión neutral o ligera sonrisa
- ❌ Sin gafas de sol
- ❌ Sin sombras fuertes

Guarda tu imagen como: `my_reference_face.jpg`

### Paso 3: Editar Configuración

Abre `example_simple.py` y modifica estas líneas:

```python
# Rutas a los modelos (ajusta según donde los descargaste)
CHECKPOINT_DIR = r"C:\models\Wan2.2-Animate-14B"
PREPROCESS_CHECKPOINT_DIR = r"C:\models\Wan2.2-Animate-Process"

# Tu imagen de referencia
REFERENCE_IMAGE = r"C:\path\to\my_reference_face.jpg"

# Configuración opcional (ajusta según tu GPU)
BATCH_SIZE = 8  # Reduce a 4 o 6 si tienes poca VRAM
WIDTH = 1280    # Reduce a 960 si tienes poca VRAM
HEIGHT = 720    # Reduce a 540 si tienes poca VRAM
```

### Paso 4: Ejecutar

```powershell
cd C:\dev\Wan2.2\realtime_face_swap
python example_simple.py
```

### Paso 5: Usar la Aplicación

Una vez iniciada:

1. **Espera**: El primer frame tardará 10-30 segundos en procesarse (cargando modelo)
2. **Observa**: Verás tu rostro reemplazado por la imagen de referencia
3. **Controla**:
   - `ESPACIO`: Pausar/Reanudar
   - `O`: Ver original vs procesado
   - `S`: Mostrar/ocultar estadísticas
   - `Q`: Salir

## ⚙️ Configuraciones Según tu GPU

### RTX 4090 / A100 (24GB VRAM)
```python
BATCH_SIZE = 8
WIDTH = 1280
HEIGHT = 720
```

### RTX 3090 / 3080 Ti (24GB / 12GB VRAM)
```python
BATCH_SIZE = 6
WIDTH = 1280
HEIGHT = 720
```

### RTX 3080 / 3070 (10GB / 8GB VRAM)
```python
BATCH_SIZE = 4
WIDTH = 960
HEIGHT = 540
```

### RTX 3060 (12GB VRAM)
```python
BATCH_SIZE = 4
WIDTH = 960
HEIGHT = 540
```

## 🐛 Solución Rápida de Problemas

### "CUDA out of memory"
```python
# En example_simple.py, reduce:
BATCH_SIZE = 2    # Reduce más
WIDTH = 640       # Reduce resolución
HEIGHT = 480
```

### "Camera not found"
```python
# Prueba diferentes IDs de cámara:
CAMERA_ID = 0  # Cámara predeterminada
CAMERA_ID = 1  # Segunda cámara
CAMERA_ID = 2  # Tercera cámara
```

### Latencia muy alta
- Normal: 2-5 segundos por lote
- Si es mayor: Reduce `BATCH_SIZE` y resolución
- Cierra otras aplicaciones que usen GPU

### Calidad del face swap baja
- Usa mejor imagen de referencia
- Aumenta resolución si tu GPU lo permite
- Mejora la iluminación de tu webcam

## 📊 ¿Qué Esperar?

**Métricas Típicas:**
- **Capture FPS**: 30 fps (tu webcam)
- **Process FPS**: 2-5 fps (modelo procesando)
- **Display FPS**: 30 fps (visualización)
- **Latency**: 2-5 segundos (tiempo de procesamiento por lote)

**Explicación:**
El sistema captura a 30fps, pero procesa en lotes cada 2-5 segundos. Los frames procesados se muestran a 30fps, creando una experiencia "semi-realtime" con latencia de unos segundos.

## 🎯 Consejos para Mejores Resultados

1. **Iluminación**: Asegúrate de tener buena luz en tu rostro
2. **Fondo**: Un fondo simple ayuda
3. **Posición**: Mantente centrado frente a la cámara
4. **Movimientos**: Movimientos lentos funcionan mejor
5. **Referencia**: Usa una foto de alta calidad como referencia

## 📞 ¿Necesitas Más Ayuda?

- Revisa el [README completo](README.md)
- Busca en los Issues de GitHub
- Lee la documentación de [Wan2.2](https://github.com/Wan-Video/Wan2.2)

---

**¡Listo para probar tu Real-Time Face Swap!** 🎭✨
