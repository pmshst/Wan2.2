# Copyright 2024-2025 Real-time Face Swap Extension for Wan2.2
"""
Script de ejemplo simplificado para probar el sistema de real-time face swap.
"""

import os
import sys

# Agregar directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.configs.wan_animate_14B import get_config
from realtime_app import RealtimeFaceSwapApp


def main():
    """
    Ejemplo simple de uso del sistema de real-time face swap.
    """
    
    # ========================================
    # CONFIGURACIÓN - MODIFICA ESTOS VALORES
    # ========================================
    
    # Rutas a los checkpoints (DEBES DESCARGAR ESTOS MODELOS)
    CHECKPOINT_DIR = r"C:\models\Wan2.2-Animate-14B"
    PREPROCESS_CHECKPOINT_DIR = r"C:\models\Wan2.2-Animate-Process"
    
    # Imagen de referencia para face swap
    # Puede ser cualquier foto con una cara clara y bien iluminada
    REFERENCE_IMAGE = r"..\examples\wan_animate\replace\image.jpeg"
    
    # Configuración de webcam
    CAMERA_ID = 0  # 0 = cámara predeterminada
    WIDTH = 1280   # Ancho de captura
    HEIGHT = 720   # Alto de captura
    FPS = 30       # FPS objetivo
    
    # Configuración de procesamiento
    BATCH_SIZE = 8  # Número de frames a procesar juntos
    DEVICE_ID = 0   # ID de la GPU (0 para la primera GPU)
    
    # ========================================
    # FIN DE CONFIGURACIÓN
    # ========================================
    
    print("="*70)
    print("  Real-Time Face Swap con Wan2.2 - Script de Ejemplo")
    print("="*70)
    print()
    print("Configuración:")
    print(f"  - Checkpoints: {CHECKPOINT_DIR}")
    print(f"  - Preprocesamiento: {PREPROCESS_CHECKPOINT_DIR}")
    print(f"  - Imagen de referencia: {REFERENCE_IMAGE}")
    print(f"  - Cámara: ID {CAMERA_ID} @ {WIDTH}x{HEIGHT} {FPS}fps")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - GPU: {DEVICE_ID}")
    print()
    
    # Validar que existan los directorios
    if not os.path.exists(CHECKPOINT_DIR):
        print("❌ ERROR: No se encontró el directorio de checkpoints del modelo")
        print(f"   Ruta: {CHECKPOINT_DIR}")
        print()
        print("Por favor descarga el modelo desde:")
        print("  https://huggingface.co/Wan-AI/Wan2.2-Animate-14B")
        print()
        return
    
    if not os.path.exists(PREPROCESS_CHECKPOINT_DIR):
        print("❌ ERROR: No se encontró el directorio de checkpoints de preprocesamiento")
        print(f"   Ruta: {PREPROCESS_CHECKPOINT_DIR}")
        print()
        print("Por favor descarga los checkpoints desde:")
        print("  https://huggingface.co/Wan-AI/Wan2.2-Animate-Process")
        print()
        return
    
    if not os.path.exists(REFERENCE_IMAGE):
        print("❌ ERROR: No se encontró la imagen de referencia")
        print(f"   Ruta: {REFERENCE_IMAGE}")
        print()
        print("Por favor proporciona una imagen de referencia válida")
        print("La imagen debe contener una cara clara y bien iluminada")
        print()
        return
    
    print("✅ Todos los archivos necesarios encontrados")
    print()
    print("="*70)
    print()
    
    try:
        # Cargar configuración del modelo
        print("Cargando configuración del modelo...")
        config = get_config()
        
        # Crear instancia de la aplicación
        print("Creando aplicación...")
        app = RealtimeFaceSwapApp(
            config=config,
            checkpoint_dir=CHECKPOINT_DIR,
            preprocess_checkpoint_dir=PREPROCESS_CHECKPOINT_DIR,
            reference_image_path=REFERENCE_IMAGE,
            camera_id=CAMERA_ID,
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            batch_size=BATCH_SIZE,
            device_id=DEVICE_ID
        )
        
        # Ejecutar aplicación
        print()
        print("="*70)
        print("  ¡Iniciando aplicación!")
        print("="*70)
        print()
        print("CONTROLES:")
        print("  - ESPACIO: Pausar/Reanudar")
        print("  - O: Toggle Original/Procesado")
        print("  - S: Toggle Estadísticas")
        print("  - Q/ESC: Salir")
        print()
        print("NOTA: El primer frame tardará en procesarse (cargando modelo).")
        print("      Después de eso, verás el face swap en acción.")
        print()
        print("="*70)
        print()
        
        app.run()
        
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n¡Hasta luego!")


if __name__ == "__main__":
    main()
