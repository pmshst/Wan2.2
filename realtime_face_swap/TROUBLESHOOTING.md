# Troubleshooting - Real-Time Face Swap

Guía completa de solución de problemas para el sistema de Real-Time Face Swap.

## 📋 Tabla de Contenidos

1. [Problemas de GPU y VRAM](#problemas-de-gpu-y-vram)
2. [Problemas de Webcam](#problemas-de-webcam)
3. [Problemas de Rendimiento](#problemas-de-rendimiento)
4. [Problemas de Calidad](#problemas-de-calidad)
5. [Errores de Instalación](#errores-de-instalación)
6. [Otros Problemas](#otros-problemas)

---

## Problemas de GPU y VRAM

### Error: "CUDA out of memory"

**Síntomas:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Soluciones:**

1. **Reducir batch_size** (más efectivo)
   ```python
   # En example_simple.py o config.ini
   BATCH_SIZE = 4  # o incluso 2
   ```

2. **Reducir clip_len** (debe ser 4n+1)
   ```python
   CLIP_LEN = 17  # o 13, 21, etc.
   ```

3. **Reducir resolución**
   ```python
   WIDTH = 960
   HEIGHT = 540
   ```

4. **Habilitar model offload** (más lento pero usa menos VRAM)
   ```python
   offload_model = True
   ```

5. **Cerrar otras aplicaciones que usen GPU**
   - Cierra navegadores con aceleración GPU
   - Cierra otros programas de IA
   - Verifica con `nvidia-smi`

6. **Usar menos sampling_steps**
   ```python
   # En face_swap_processor.py, línea ~180
   sampling_steps = 10  # en vez de 15
   ```

### Error: "No CUDA-capable device detected"

**Síntomas:**
```
RuntimeError: No CUDA-capable device is detected
```

**Soluciones:**

1. **Verificar instalación de CUDA**
   ```powershell
   nvcc --version
   nvidia-smi
   ```

2. **Reinstalar PyTorch con CUDA**
   ```powershell
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verificar drivers de NVIDIA**
   - Actualiza a la versión más reciente
   - Descarga desde [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)

### GPU No Detectada

**Síntomas:**
- nvidia-smi falla
- Sistema usa CPU en lugar de GPU

**Soluciones:**

1. **Reinstalar drivers de NVIDIA**
2. **Verificar en Device Manager** (Administrador de Dispositivos)
3. **Reiniciar el sistema**

---

## Problemas de Webcam

### Error: "Camera not found" o "Error al abrir la cámara"

**Soluciones:**

1. **Probar diferentes IDs de cámara**
   ```python
   CAMERA_ID = 0  # Prueba 0, 1, 2, 3...
   ```

2. **Verificar que la cámara no esté en uso**
   - Cierra Zoom, Teams, Skype, etc.
   - Cierra navegadores que puedan usar la cámara

3. **Verificar permisos en Windows**
   - Configuración → Privacidad → Cámara
   - Asegúrate de que las aplicaciones puedan acceder a la cámara

4. **Test de cámara con OpenCV**
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   print(cap.isOpened())
   cap.release()
   ```

### Webcam con Baja Calidad

**Soluciones:**

1. **Mejorar iluminación**
   - Usa luz frontal
   - Evita contraluz

2. **Limpiar lente de la cámara**

3. **Ajustar configuración de la cámara**
   - En Windows: Configuración de Cámara
   - Aumentar brillo/contraste si es necesario

### FPS de Captura Bajo

**Síntomas:**
- "Capture FPS" muestra valores bajos (<20)

**Soluciones:**

1. **Reducir resolución de captura**
   ```python
   WIDTH = 960
   HEIGHT = 540
   ```

2. **Cerrar otras aplicaciones que usen la cámara**

3. **Verificar que la cámara soporte la resolución/fps configurados**

---

## Problemas de Rendimiento

### Latencia Muy Alta (>10 segundos)

**Causas posibles:**
- GPU demasiado lenta
- Batch size muy grande
- Resolución muy alta
- Otros procesos usando GPU

**Soluciones:**

1. **Reducir batch_size**
   ```python
   BATCH_SIZE = 2  # Procesa con más frecuencia
   ```

2. **Reducir clip_len**
   ```python
   CLIP_LEN = 13  # Mínimo recomendado: 13 (4*3+1)
   ```

3. **Reducir resolución**
   ```python
   WIDTH = 640
   HEIGHT = 480
   ```

4. **Verificar uso de GPU**
   ```powershell
   nvidia-smi -l 1  # Monitorear cada segundo
   ```

5. **Cerrar procesos en GPU**
   - Usa Task Manager para ver qué usa la GPU
   - Cierra aplicaciones innecesarias

### Process FPS Muy Bajo (<1)

**Soluciones:**

1. **Verificar que se use GPU, no CPU**
   - Revisa en las estadísticas que diga "cuda:0"

2. **Optimizar configuración según tu GPU**
   - Consulta la sección [Performance] en config.ini

3. **Reducir sampling_steps**
   ```python
   sampling_steps = 8  # Mínimo: 8, recomendado: 10-15
   ```

### Aplicación Se Congela

**Soluciones:**

1. **Verificar que los threads estén funcionando**
   - Error en captura o procesamiento puede congelar todo

2. **Revisar logs/errores en consola**

3. **Reiniciar la aplicación**

4. **Reducir carga del sistema**

---

## Problemas de Calidad

### Face Swap de Mala Calidad

**Causas posibles:**
- Imagen de referencia de baja calidad
- Iluminación pobre
- Configuración subóptima

**Soluciones:**

1. **Mejorar imagen de referencia**
   - Usa foto de alta resolución (>512x512)
   - Cara frontal y centrada
   - Buena iluminación
   - Sin gafas de sol o sombras fuertes

2. **Aumentar sampling_steps**
   ```python
   sampling_steps = 20  # Mejor calidad, más lento
   ```

3. **Aumentar resolución**
   ```python
   WIDTH = 1280
   HEIGHT = 720
   ```

4. **Mejorar iluminación de webcam**

### Parpadeo o Inconsistencias Entre Frames

**Soluciones:**

1. **Aumentar refert_num**
   ```python
   refert_num = 5  # Más consistencia temporal (en vez de 1)
   ```

2. **Aumentar clip_len**
   ```python
   CLIP_LEN = 33  # Más contexto temporal
   ```

3. **Usar guide_scale** (más lento)
   ```python
   guide_scale = 2.0  # Más control, más lento
   ```

### Colores Incorrectos o Saturación

**Soluciones:**

1. **Verificar conversión RGB/BGR**
   - El código ya maneja esto, pero verifica si modificaste algo

2. **Ajustar iluminación**

3. **Probar con diferente imagen de referencia**

---

## Errores de Instalación

### Error al Instalar requirements.txt

**Soluciones:**

1. **Instalar en orden**
   ```powershell
   # Primero torch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # Luego el resto
   pip install -r requirements_windows.txt
   ```

2. **Usar entorno virtual**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements_windows.txt
   ```

### Error: "flash_attn" No Se Puede Instalar

**Soluciones:**

1. **Instalar otras dependencias primero**
   ```powershell
   # flash_attn es opcional, instala lo demás primero
   ```

2. **Skip flash_attn si falla**
   - La aplicación puede funcionar sin él
   - Será más lenta pero funcional

### ModuleNotFoundError

**Soluciones:**

1. **Verificar que estás en el directorio correcto**
   ```powershell
   cd C:\dev\Wan2.2\realtime_face_swap
   ```

2. **Verificar instalación de dependencias**
   ```powershell
   pip list | Select-String "torch"
   pip list | Select-String "opencv"
   ```

3. **Reinstalar paquete específico**
   ```powershell
   pip install --force-reinstall nombre_paquete
   ```

---

## Otros Problemas

### Aplicación No Responde

**Soluciones:**

1. **Ctrl+C** en la terminal para interrumpir

2. **Verificar logs en consola**

3. **Reiniciar aplicación**

4. **Reducir carga (batch_size, resolución)**

### Error: "Access Denied" o Permisos

**Soluciones:**

1. **Ejecutar como Administrador** (si es necesario)

2. **Verificar permisos de archivos/directorios**

3. **Verificar permisos de cámara en Windows**

### Error: "Cannot Import Name..."

**Soluciones:**

1. **Verificar estructura de archivos**
   ```powershell
   python test_system.py
   ```

2. **Reinstalar el paquete**

3. **Verificar que todos los archivos .py estén presentes**

### Ventana OpenCV No Se Cierra

**Soluciones:**

1. **Presionar Q o ESC**

2. **Forzar cierre con Ctrl+C en terminal**

3. **Cerrar desde Task Manager si es necesario**

---

## 🔍 Herramientas de Diagnóstico

### Verificar Sistema

```powershell
# Ejecutar script de verificación
.\setup_and_verify.ps1
```

### Verificar GPU

```powershell
# Información de GPU
nvidia-smi

# Monitoreo continuo
nvidia-smi -l 1
```

### Test de Importaciones

```powershell
python -c "import torch; print(torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__)"
```

### Ejecutar Tests

```powershell
python test_system.py
```

---

## 📞 Obtener Ayuda Adicional

Si ninguna de estas soluciones funciona:

1. **Revisa los logs en consola** - Contienen información detallada del error

2. **Ejecuta el test de sistema**
   ```powershell
   python test_system.py
   ```

3. **Verifica configuración de hardware**
   - GPU compatible
   - Drivers actualizados
   - Suficiente VRAM

4. **Busca en Issues de GitHub**
   - [Wan2.2 Issues](https://github.com/Wan-Video/Wan2.2/issues)

5. **Crea un nuevo Issue con**
   - Descripción del problema
   - Mensaje de error completo
   - Configuración de hardware
   - Pasos para reproducir

---

## 📊 Configuraciones Probadas

### Configuración Mínima Funcional
```
GPU: RTX 3060 (12GB)
batch_size: 2
clip_len: 13
resolution: 640x480
sampling_steps: 8
Latencia: ~8-10s
```

### Configuración Recomendada
```
GPU: RTX 4090 (24GB)
batch_size: 8
clip_len: 25
resolution: 1280x720
sampling_steps: 15
Latency: ~2-4s
```

### Configuración de Alta Calidad
```
GPU: A100 (40GB)
batch_size: 12
clip_len: 33
resolution: 1920x1080
sampling_steps: 20
Latency: ~5-8s
```

---

**¿Encontraste una solución que no está aquí?** Contribuye al proyecto agregándola a este documento!
