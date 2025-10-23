# Arquitectura del Sistema Real-Time Face Swap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME FACE SWAP SYSTEM                            │
│                     Powered by Wan2.2-Animate                            │
└─────────────────────────────────────────────────────────────────────────┘

                                                                             
┌─────────────────────────────────────────────────────────────────────────┐
│  CAPA 1: CAPTURA (webcam_capture.py)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐        ┌─────────────────┐      ┌──────────────┐    │
│  │   Webcam     │ ──────>│ Capture Thread  │─────>│ Frame Buffer │    │
│  │              │  30fps  │  (Threading)    │      │  (Circular)  │    │
│  └──────────────┘        └─────────────────┘      └──────────────┘    │
│                                                           │              │
│                                                           │ Latest       │
│                                                           │ Frames       │
└───────────────────────────────────────────────────────────┼──────────────┘
                                                            │
                                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CAPA 2: COORDINACIÓN (realtime_app.py)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    RealtimeFaceSwapApp                           │  │
│  │                                                                    │  │
│  │  ┌────────────────┐      ┌─────────────────┐   ┌─────────────┐ │  │
│  │  │ Feeder Thread  │─────>│ Frame Queue     │──>│  Processor  │ │  │
│  │  │ (100 Hz)       │      │ (Input Buffer)  │   │  (Async)    │ │  │
│  │  └────────────────┘      └─────────────────┘   └─────────────┘ │  │
│  │                                                        │          │  │
│  │                                                        │          │  │
│  │  ┌────────────────┐                                  │          │  │
│  │  │ Display Thread │<─────────────────────────────────┘          │  │
│  │  │ (30 fps)       │      Processed Frames                       │  │
│  │  └────────────────┘                                             │  │
│  │         │                                                         │  │
│  │         ▼                                                         │  │
│  │  ┌────────────────┐                                             │  │
│  │  │ UI Overlay     │  Stats, Controls, Info                      │  │
│  │  └────────────────┘                                             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CAPA 3: PROCESAMIENTO (face_swap_processor.py)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              AsyncFaceSwapProcessor                              │  │
│  │                                                                    │  │
│  │  ┌────────────────┐                                              │  │
│  │  │ Processing     │  Loop ejecutándose en thread separado        │  │
│  │  │ Thread         │                                              │  │
│  │  └────────┬───────┘                                              │  │
│  │           │                                                       │  │
│  │           │ Obtiene frames en lotes (batch_size)                │  │
│  │           ▼                                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │          RealtimeFaceSwap                                   │ │  │
│  │  │                                                              │ │  │
│  │  │  1. Preprocess Frames                                       │ │  │
│  │  │     - Detección de pose (Pose2d)                           │ │  │
│  │  │     - Detección facial (face detection)                     │ │  │
│  │  │     - Generación de máscaras (SAM2)                        │ │  │
│  │  │     - Extracción de features                                │ │  │
│  │  │                                                              │ │  │
│  │  │  2. Face Swap Generation                                    │ │  │
│  │  │     - Encode frames con VAE                                 │ │  │
│  │  │     - Encode referencia con CLIP                           │ │  │
│  │  │     - Generate con Wan2.2 DiT                              │ │  │
│  │  │     - Decode con VAE                                        │ │  │
│  │  │                                                              │ │  │
│  │  │  3. Post-processing                                         │ │  │
│  │  │     - Desnormalización                                      │ │  │
│  │  │     - Conversión de formato                                 │ │  │
│  │  │     - Buffer de salida                                      │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CAPA 4: MODELOS DE IA (Wan2.2 Core)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Preprocessing Models                                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │
│  │  │   Pose2D    │  │    Face     │  │   SAM2 (Masking)    │   │   │
│  │  │  Detection  │  │  Detection  │  │                     │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Wan2.2-Animate-14B                                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │
│  │  │ T5 Encoder  │  │    CLIP     │  │    VAE 2.1          │   │   │
│  │  │   (Text)    │  │  (Vision)   │  │   (Encoder/Dec)     │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │   │
│  │                                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │        DiT (Diffusion Transformer) - MoE                │  │   │
│  │  │  - 14B parameters                                       │  │   │
│  │  │  - Temporal attention                                   │  │   │
│  │  │  - Face conditioning                                    │  │   │
│  │  │  - Motion control                                       │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
 FLUJO DE DATOS
═══════════════════════════════════════════════════════════════════════════

Captura (30 fps)
    │
    ├─> Frame Buffer (circular, 30 frames)
    │
    └─> Feeder Thread (100 Hz)
            │
            └─> Input Queue (60 frames max)
                    │
                    └─> Processing Thread
                            │
                            ├─> Acumula hasta batch_size frames
                            │
                            ├─> Preprocesa lote
                            │   ├─> Detección de pose
                            │   ├─> Detección facial
                            │   └─> Generación de máscaras
                            │
                            ├─> Face Swap con Wan2.2
                            │   ├─> Encode con VAE
                            │   ├─> Difussion con DiT
                            │   └─> Decode con VAE
                            │
                            └─> Output Buffer (60 frames)
                                    │
                                    └─> Display Thread (30 fps)
                                            │
                                            └─> Ventana OpenCV


═══════════════════════════════════════════════════════════════════════════
 MÉTRICAS DE RENDIMIENTO TÍPICAS
═══════════════════════════════════════════════════════════════════════════

RTX 4090 (24GB):
  - Capture FPS: 30
  - Process FPS: 3-5 (frames procesados/segundo)
  - Display FPS: 30
  - Latency: 2-4 segundos/lote
  - Batch Size: 8 frames
  - Resolution: 1280x720

RTX 3090 (24GB):
  - Capture FPS: 30
  - Process FPS: 2-4
  - Display FPS: 30
  - Latency: 3-5 segundos/lote
  - Batch Size: 6 frames
  - Resolution: 1280x720

RTX 3080 (10GB):
  - Capture FPS: 30
  - Process FPS: 1-2
  - Display FPS: 30
  - Latency: 4-8 segundos/lote
  - Batch Size: 4 frames
  - Resolution: 960x540


═══════════════════════════════════════════════════════════════════════════
 OPTIMIZACIONES IMPLEMENTADAS
═══════════════════════════════════════════════════════════════════════════

1. THREADING
   - Captura en thread separado (no bloquea)
   - Procesamiento en thread separado (no bloquea UI)
   - Display en thread principal (responsive)

2. BUFFERING
   - Buffer circular para captura (evita pérdida de frames)
   - Cola de entrada para procesamiento
   - Buffer de salida para display smooth

3. BATCH PROCESSING
   - Procesa múltiples frames juntos
   - Mejor utilización de GPU
   - Reduce overhead de llamadas al modelo

4. CACHING
   - Imagen de referencia pre-cargada
   - Modelos cargados una sola vez
   - Reutilización de datos preprocesados

5. MEMORY MANAGEMENT
   - Límites en tamaños de buffers
   - Limpieza de recursos temporales
   - Offload opcional de modelos


═══════════════════════════════════════════════════════════════════════════
 LIMITACIONES Y TRADE-OFFS
═══════════════════════════════════════════════════════════════════════════

LIMITACIONES:
  - NO es verdadero "real-time" (60fps sin latencia)
  - Latencia de 2-5 segundos es inevitable con modelos pesados
  - Requiere GPU potente (mínimo 16GB VRAM)
  - Calidad vs velocidad son inversamente proporcionales

TRADE-OFFS:
  - Mayor batch_size = Mayor throughput pero mayor latencia
  - Mayor resolución = Mejor calidad pero menor velocidad
  - Más sampling_steps = Mejor calidad pero más lento
  - Offload modelos = Menos VRAM pero mucho más lento


═══════════════════════════════════════════════════════════════════════════
```

## Componentes Principales

### 1. WebcamCapture
- **Propósito**: Captura frames de webcam sin bloqueo
- **Tecnología**: OpenCV + Threading
- **Características**:
  - Buffer circular de 30 frames
  - Control de FPS
  - Estadísticas en tiempo real

### 2. RealtimeFaceSwap
- **Propósito**: Procesamiento de face swap
- **Tecnología**: Wan2.2-Animate + PyTorch
- **Características**:
  - Preprocesamiento integrado
  - Procesamiento por lotes
  - Caché de referencia

### 3. AsyncFaceSwapProcessor
- **Propósito**: Gestión asíncrona de procesamiento
- **Tecnología**: Threading + Queue
- **Características**:
  - No bloquea captura ni display
  - Acumulación de frames en lotes
  - Buffer de salida

### 4. RealtimeFaceSwapApp
- **Propósito**: Coordinación y UI
- **Tecnología**: OpenCV + Threading
- **Características**:
  - Gestión de múltiples threads
  - Overlay de información
  - Controles interactivos

## Tecnologías Utilizadas

- **Python 3.10+**: Lenguaje principal
- **PyTorch**: Framework de deep learning
- **OpenCV**: Captura y visualización de video
- **Wan2.2-Animate**: Modelo de face swap
- **Threading**: Procesamiento asíncrono
- **NumPy**: Operaciones numéricas

## Referencias

- [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2-Animate Paper](https://humanaigc.github.io/wan-animate)
- [OpenCV Documentation](https://docs.opencv.org/)
