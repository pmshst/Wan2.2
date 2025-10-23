# Copyright 2024-2025 Real-time Face Swap Extension for Wan2.2
"""
Sistema principal de Real-Time Face Swap con interfaz de usuario.
Integra captura de webcam, procesamiento y visualización en tiempo real.
"""

import os
import cv2
import numpy as np
import threading
import time
from typing import Optional
import argparse

from webcam_capture import WebcamCapture, FramePreprocessor
from face_swap_processor import RealtimeFaceSwap, AsyncFaceSwapProcessor


class RealtimeFaceSwapApp:
    """
    Aplicación principal de Real-Time Face Swap.
    """
    
    def __init__(
        self,
        config,
        checkpoint_dir: str,
        preprocess_checkpoint_dir: str,
        reference_image_path: str,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        batch_size: int = 8,
        device_id: int = 0
    ):
        """
        Inicializa la aplicación.
        
        Args:
            config: Configuración del modelo Wan2.2
            checkpoint_dir: Directorio de checkpoints del modelo
            preprocess_checkpoint_dir: Directorio de checkpoints de preprocesamiento
            reference_image_path: Imagen de referencia para face swap
            camera_id: ID de la cámara
            width: Ancho de captura
            height: Alto de captura
            fps: FPS objetivo
            batch_size: Tamaño del lote para procesamiento
            device_id: ID del dispositivo GPU
        """
        print("="*60)
        print("Inicializando Real-Time Face Swap con Wan2.2")
        print("="*60)
        
        # Parámetros
        self.width = width
        self.height = height
        self.fps = fps
        
        # Inicializar captura de webcam
        print("\n[1/3] Inicializando captura de webcam...")
        self.webcam = WebcamCapture(
            camera_id=camera_id,
            width=width,
            height=height,
            fps=fps,
            buffer_size=30
        )
        
        # Inicializar procesador de face swap
        print("\n[2/3] Inicializando procesador de Face Swap...")
        self.face_swap = RealtimeFaceSwap(
            config=config,
            checkpoint_dir=checkpoint_dir,
            preprocess_checkpoint_dir=preprocess_checkpoint_dir,
            reference_image_path=reference_image_path,
            device_id=device_id,
            batch_size=batch_size,
            clip_len=25,
            use_gpu=True,
            offload_model=False
        )
        
        # Inicializar procesador asíncrono
        print("\n[3/3] Inicializando procesamiento asíncrono...")
        self.async_processor = AsyncFaceSwapProcessor(
            face_swap_processor=self.face_swap,
            batch_size=batch_size
        )
        
        # Estado de la aplicación
        self.is_running = False
        self.show_original = False
        self.show_stats = True
        self.paused = False
        
        # Thread para envío de frames al procesador
        self.feeder_thread = None
        
        # Estadísticas
        self.display_fps = 0.0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        print("\n" + "="*60)
        print("Inicialización completada")
        print("="*60)
    
    def _frame_feeder_loop(self):
        """
        Loop que alimenta frames al procesador (ejecutado en thread separado).
        """
        while self.is_running:
            if not self.paused:
                # Obtener frame de la webcam
                frame = self.webcam.get_latest_frame()
                
                if frame is not None:
                    # Enviar al procesador asíncrono
                    self.async_processor.add_frame(frame)
            
            time.sleep(0.01)  # 100 Hz
    
    def _draw_overlay(self, frame: np.ndarray, is_processed: bool = False) -> np.ndarray:
        """
        Dibuja información superpuesta en el frame.
        
        Args:
            frame: Frame sobre el que dibujar
            is_processed: Si es un frame procesado o original
            
        Returns:
            Frame con overlay
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        if not self.show_stats:
            return overlay
        
        # Fondo semi-transparente para el texto
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Información de estado
        y_offset = 35
        line_height = 25
        
        # Título
        title = "PROCESADO (Face Swap)" if is_processed else "ORIGINAL (Webcam)"
        color = (0, 255, 0) if is_processed else (0, 165, 255)
        cv2.putText(frame, title, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height
        
        # FPS de visualización
        cv2.putText(frame, f"Display FPS: {self.display_fps:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Información de webcam
        webcam_info = self.webcam.get_buffer_info()
        cv2.putText(frame, f"Capture FPS: {webcam_info['fps']:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Información de procesamiento
        proc_stats = self.face_swap.get_stats()
        cv2.putText(frame, f"Process FPS: {proc_stats['processing_fps']:.2f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Latency: {proc_stats['processing_latency']:.2f}s", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Estado
        status = "PAUSADO" if self.paused else "ACTIVO"
        status_color = (0, 165, 255) if self.paused else (0, 255, 0)
        cv2.putText(frame, f"Estado: {status}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Controles (en la parte inferior)
        controls = [
            "Controles:",
            "ESPACIO: Pausar/Reanudar",
            "O: Toggle Original/Procesado",
            "S: Toggle Estadisticas",
            "Q/ESC: Salir"
        ]
        
        y_bottom = h - 20
        for i, control in enumerate(reversed(controls)):
            y_pos = y_bottom - (i * 20)
            cv2.putText(frame, control, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def start(self):
        """
        Inicia la aplicación.
        """
        print("\nIniciando aplicación...")
        
        # Iniciar captura de webcam
        if not self.webcam.start():
            print("Error: No se pudo iniciar la webcam")
            return False
        
        # Esperar a que se capturen algunos frames
        print("Esperando frames iniciales...")
        time.sleep(1.0)
        
        # Iniciar procesador asíncrono
        self.async_processor.start()
        
        # Iniciar thread feeder
        self.is_running = True
        self.feeder_thread = threading.Thread(target=self._frame_feeder_loop, daemon=True)
        self.feeder_thread.start()
        
        print("\n" + "="*60)
        print("¡Aplicación iniciada correctamente!")
        print("="*60)
        print("\nPresiona 'Q' o 'ESC' para salir")
        print("Presiona 'O' para alternar entre original y procesado")
        print("Presiona 'ESPACIO' para pausar/reanudar")
        print("Presiona 'S' para mostrar/ocultar estadísticas")
        
        return True
    
    def run(self):
        """
        Loop principal de la aplicación.
        """
        if not self.start():
            return
        
        try:
            # Crear ventana
            window_name = "Real-Time Face Swap - Wan2.2"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.width, self.height)
            
            while self.is_running:
                # Obtener frame apropiado
                if self.show_original:
                    display_frame = self.webcam.get_latest_frame()
                else:
                    display_frame = self.face_swap.get_from_buffer()
                    
                    # Si no hay frame procesado, mostrar original
                    if display_frame is None:
                        display_frame = self.webcam.get_latest_frame()
                        if display_frame is not None:
                            # Agregar mensaje de "Procesando..."
                            h, w = display_frame.shape[:2]
                            cv2.putText(display_frame, "Procesando primer frame...", 
                                       (w//2 - 200, h//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                if display_frame is not None:
                    # Agregar overlay
                    is_processed = not self.show_original and self.face_swap.get_from_buffer() is not None
                    display_frame = self._draw_overlay(display_frame, is_processed)
                    
                    # Mostrar frame
                    cv2.imshow(window_name, display_frame)
                    
                    # Actualizar FPS de visualización
                    self.fps_counter += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.display_fps = self.fps_counter / (current_time - self.last_fps_time)
                        self.fps_counter = 0
                        self.last_fps_time = current_time
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' o ESC
                    print("\nSaliendo...")
                    break
                elif key == ord('o') or key == ord('O'):
                    self.show_original = not self.show_original
                    mode = "ORIGINAL" if self.show_original else "PROCESADO"
                    print(f"Modo cambiado a: {mode}")
                elif key == ord(' '):  # Espacio
                    self.paused = not self.paused
                    status = "PAUSADO" if self.paused else "REANUDADO"
                    print(f"Procesamiento {status}")
                elif key == ord('s') or key == ord('S'):
                    self.show_stats = not self.show_stats
                    status = "mostradas" if self.show_stats else "ocultas"
                    print(f"Estadísticas {status}")
        
        finally:
            self.stop()
    
    def stop(self):
        """
        Detiene la aplicación y libera recursos.
        """
        print("\nDeteniendo aplicación...")
        
        self.is_running = False
        
        # Esperar al thread feeder
        if self.feeder_thread:
            self.feeder_thread.join(timeout=2.0)
        
        # Detener procesador asíncrono
        self.async_processor.stop()
        
        # Detener webcam
        self.webcam.stop()
        
        # Limpiar recursos
        self.face_swap.cleanup()
        
        # Cerrar ventanas
        cv2.destroyAllWindows()
        
        print("Aplicación detenida")
    
    def __del__(self):
        """
        Destructor: asegura limpieza de recursos.
        """
        self.stop()


def main():
    """
    Función principal.
    """
    parser = argparse.ArgumentParser(description='Real-Time Face Swap con Wan2.2')
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directorio con checkpoints del modelo Wan2.2-Animate')
    parser.add_argument('--preprocess_checkpoint_dir', type=str, required=True,
                       help='Directorio con checkpoints de preprocesamiento')
    parser.add_argument('--reference_image', type=str, required=True,
                       help='Ruta a la imagen de referencia para face swap')
    parser.add_argument('--config', type=str, required=True,
                       help='Archivo de configuración del modelo')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='ID de la cámara (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Ancho de captura (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Alto de captura (default: 720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='FPS objetivo (default: 30)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Tamaño del lote para procesamiento (default: 8)')
    parser.add_argument('--device_id', type=int, default=0,
                       help='ID del dispositivo GPU (default: 0)')
    
    args = parser.parse_args()
    
    # Cargar configuración
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from wan.configs.wan_animate_14B import get_config
    
    config = get_config()
    
    # Crear y ejecutar aplicación
    app = RealtimeFaceSwapApp(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        preprocess_checkpoint_dir=args.preprocess_checkpoint_dir,
        reference_image_path=args.reference_image,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        batch_size=args.batch_size,
        device_id=args.device_id
    )
    
    app.run()


if __name__ == "__main__":
    main()
