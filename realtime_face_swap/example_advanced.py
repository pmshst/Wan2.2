# Copyright 2024-2025 Real-time Face Swap Extension for Wan2.2
"""
Script de ejemplo avanzado con configuración personalizada.
Muestra cómo usar el sistema con parámetros personalizados y configuración avanzada.
"""

import os
import sys
import argparse
import configparser
from pathlib import Path

# Agregar directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.configs.wan_animate_14B import get_config
from realtime_face_swap import RealtimeFaceSwapApp


def load_config_file(config_path: str) -> dict:
    """
    Carga configuración desde archivo .ini
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario con la configuración
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    settings = {
        # Hardware
        'device_id': config.getint('Hardware', 'device_id', fallback=0),
        'use_gpu': config.getboolean('Hardware', 'use_gpu', fallback=True),
        'offload_model': config.getboolean('Hardware', 'offload_model', fallback=False),
        
        # Webcam
        'camera_id': config.getint('Webcam', 'camera_id', fallback=0),
        'width': config.getint('Webcam', 'width', fallback=1280),
        'height': config.getint('Webcam', 'height', fallback=720),
        'fps': config.getint('Webcam', 'fps', fallback=30),
        
        # Processing
        'batch_size': config.getint('Processing', 'batch_size', fallback=8),
        'clip_len': config.getint('Processing', 'clip_len', fallback=25),
        
        # Paths
        'checkpoint_dir': config.get('Paths', 'checkpoint_dir'),
        'preprocess_checkpoint_dir': config.get('Paths', 'preprocess_checkpoint_dir'),
        'reference_image': config.get('Paths', 'reference_image'),
    }
    
    return settings


def validate_settings(settings: dict) -> tuple[bool, list]:
    """
    Valida la configuración.
    
    Args:
        settings: Diccionario con configuración
        
    Returns:
        Tupla (es_válido, lista_de_errores)
    """
    errors = []
    
    # Validar paths
    if not os.path.exists(settings['checkpoint_dir']):
        errors.append(f"Checkpoint directory no encontrado: {settings['checkpoint_dir']}")
    
    if not os.path.exists(settings['preprocess_checkpoint_dir']):
        errors.append(f"Preprocess checkpoint directory no encontrado: {settings['preprocess_checkpoint_dir']}")
    
    if not os.path.exists(settings['reference_image']):
        errors.append(f"Reference image no encontrada: {settings['reference_image']}")
    
    # Validar clip_len (debe ser 4n+1)
    if (settings['clip_len'] - 1) % 4 != 0:
        errors.append(f"clip_len debe ser 4n+1, recibido: {settings['clip_len']}")
    
    # Validar batch_size
    if settings['batch_size'] < 1:
        errors.append(f"batch_size debe ser >= 1, recibido: {settings['batch_size']}")
    
    # Validar resolución
    if settings['width'] < 640 or settings['height'] < 480:
        errors.append(f"Resolución muy baja: {settings['width']}x{settings['height']}")
    
    return len(errors) == 0, errors


def print_settings(settings: dict):
    """
    Imprime la configuración de forma legible.
    
    Args:
        settings: Diccionario con configuración
    """
    print()
    print("="*70)
    print("  CONFIGURACIÓN DEL SISTEMA")
    print("="*70)
    print()
    
    print("Hardware:")
    print(f"  - GPU ID: {settings['device_id']}")
    print(f"  - Usar GPU: {'Sí' if settings['use_gpu'] else 'No'}")
    print(f"  - Offload Model: {'Sí' if settings['offload_model'] else 'No'}")
    print()
    
    print("Webcam:")
    print(f"  - Camera ID: {settings['camera_id']}")
    print(f"  - Resolución: {settings['width']}x{settings['height']}")
    print(f"  - FPS: {settings['fps']}")
    print()
    
    print("Procesamiento:")
    print(f"  - Batch Size: {settings['batch_size']}")
    print(f"  - Clip Length: {settings['clip_len']}")
    print()
    
    print("Rutas:")
    print(f"  - Checkpoints: {settings['checkpoint_dir']}")
    print(f"  - Preprocess: {settings['preprocess_checkpoint_dir']}")
    print(f"  - Referencia: {settings['reference_image']}")
    print()
    print("="*70)
    print()


def main():
    """
    Función principal del ejemplo avanzado.
    """
    parser = argparse.ArgumentParser(
        description='Real-Time Face Swap - Ejemplo Avanzado',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  1. Con archivo de configuración:
     python example_advanced.py --config config.ini

  2. Con parámetros de línea de comandos:
     python example_advanced.py \\
         --checkpoint_dir "C:/models/Wan2.2-Animate-14B" \\
         --preprocess_checkpoint_dir "C:/models/Wan2.2-Animate-Process" \\
         --reference_image "reference.jpg" \\
         --batch_size 8 --width 1280 --height 720

  3. Combinado (config.ini + override):
     python example_advanced.py --config config.ini --batch_size 4
        """
    )
    
    # Argumentos
    parser.add_argument('--config', type=str, default='config.ini',
                       help='Archivo de configuración .ini')
    
    # Override de configuración
    parser.add_argument('--checkpoint_dir', type=str,
                       help='Directorio de checkpoints (override)')
    parser.add_argument('--preprocess_checkpoint_dir', type=str,
                       help='Directorio de checkpoints de preprocesamiento (override)')
    parser.add_argument('--reference_image', type=str,
                       help='Imagen de referencia (override)')
    
    parser.add_argument('--camera_id', type=int,
                       help='ID de la cámara (override)')
    parser.add_argument('--width', type=int,
                       help='Ancho de captura (override)')
    parser.add_argument('--height', type=int,
                       help='Alto de captura (override)')
    parser.add_argument('--fps', type=int,
                       help='FPS de captura (override)')
    
    parser.add_argument('--batch_size', type=int,
                       help='Tamaño del lote (override)')
    parser.add_argument('--clip_len', type=int,
                       help='Longitud del clip (override)')
    parser.add_argument('--device_id', type=int,
                       help='ID del dispositivo GPU (override)')
    
    parser.add_argument('--offload_model', action='store_true',
                       help='Habilitar model offload (override)')
    
    parser.add_argument('--no_validation', action='store_true',
                       help='Saltar validación de configuración')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Salida verbose con más información')
    
    args = parser.parse_args()
    
    # Banner
    print()
    print("="*70)
    print("  REAL-TIME FACE SWAP - EJEMPLO AVANZADO")
    print("  Powered by Wan2.2-Animate")
    print("="*70)
    
    # Cargar configuración desde archivo
    config_file = Path(__file__).parent / args.config
    
    if config_file.exists():
        print(f"\n✓ Cargando configuración desde: {config_file}")
        settings = load_config_file(str(config_file))
    else:
        if args.checkpoint_dir and args.preprocess_checkpoint_dir and args.reference_image:
            print("\n⚠ Archivo de configuración no encontrado, usando parámetros de línea de comandos")
            settings = {
                'device_id': 0,
                'use_gpu': True,
                'offload_model': False,
                'camera_id': 0,
                'width': 1280,
                'height': 720,
                'fps': 30,
                'batch_size': 8,
                'clip_len': 25,
                'checkpoint_dir': '',
                'preprocess_checkpoint_dir': '',
                'reference_image': '',
            }
        else:
            print(f"\n❌ ERROR: Archivo de configuración no encontrado: {config_file}")
            print("   Y no se proporcionaron todos los parámetros requeridos.")
            print()
            print("   Crea un config.ini o proporciona:")
            print("   --checkpoint_dir, --preprocess_checkpoint_dir, --reference_image")
            return 1
    
    # Aplicar overrides de línea de comandos
    if args.checkpoint_dir:
        settings['checkpoint_dir'] = args.checkpoint_dir
    if args.preprocess_checkpoint_dir:
        settings['preprocess_checkpoint_dir'] = args.preprocess_checkpoint_dir
    if args.reference_image:
        settings['reference_image'] = args.reference_image
    if args.camera_id is not None:
        settings['camera_id'] = args.camera_id
    if args.width:
        settings['width'] = args.width
    if args.height:
        settings['height'] = args.height
    if args.fps:
        settings['fps'] = args.fps
    if args.batch_size:
        settings['batch_size'] = args.batch_size
    if args.clip_len:
        settings['clip_len'] = args.clip_len
    if args.device_id is not None:
        settings['device_id'] = args.device_id
    if args.offload_model:
        settings['offload_model'] = True
    
    # Mostrar configuración
    print_settings(settings)
    
    # Validar configuración
    if not args.no_validation:
        print("Validando configuración...")
        is_valid, errors = validate_settings(settings)
        
        if not is_valid:
            print("\n❌ ERRORES EN LA CONFIGURACIÓN:")
            for error in errors:
                print(f"   - {error}")
            print()
            return 1
        
        print("✓ Configuración validada correctamente")
    
    # Confirmar inicio
    print()
    print("="*70)
    print("  ¿Iniciar aplicación con esta configuración?")
    print("="*70)
    print()
    
    if not args.verbose:
        confirm = input("Presiona ENTER para continuar o Ctrl+C para cancelar...")
    
    try:
        # Cargar configuración del modelo
        print("\nCargando configuración del modelo Wan2.2...")
        config = get_config()
        
        # Crear aplicación
        print("Creando aplicación...")
        app = RealtimeFaceSwapApp(
            config=config,
            checkpoint_dir=settings['checkpoint_dir'],
            preprocess_checkpoint_dir=settings['preprocess_checkpoint_dir'],
            reference_image_path=settings['reference_image'],
            camera_id=settings['camera_id'],
            width=settings['width'],
            height=settings['height'],
            fps=settings['fps'],
            batch_size=settings['batch_size'],
            device_id=settings['device_id']
        )
        
        # Ejecutar
        print()
        print("="*70)
        print("  INICIANDO APLICACIÓN")
        print("="*70)
        print()
        print("CONTROLES:")
        print("  - ESPACIO: Pausar/Reanudar")
        print("  - O: Toggle Original/Procesado")
        print("  - S: Toggle Estadísticas")
        print("  - Q/ESC: Salir")
        print()
        print("="*70)
        print()
        
        app.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrumpido por el usuario")
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
