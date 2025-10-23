# Copyright 2024-2025 Real-time Face Swap Extension for Wan2.2
"""
Módulo de captura de video en tiempo real desde webcam.
Implementa un buffer circular para gestionar frames eficientemente.
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
from typing import Optional, Tuple


class WebcamCapture:
    """
    Captura frames de webcam en tiempo real con gestión de buffer.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 30
    ):
        """
        Inicializa el capturador de webcam.
        
        Args:
            camera_id: ID de la cámara (0 para cámara predeterminada)
            width: Ancho de captura
            height: Alto de captura
            fps: Frames por segundo deseados
            buffer_size: Tamaño del buffer circular
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        # Buffer circular para frames
        self.frame_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        
        # Estado de captura
        self.is_capturing = False
        self.capture_thread = None
        self.cap = None
        
        # Estadísticas
        self.frames_captured = 0
        self.fps_actual = 0.0
        self.last_fps_update = time.time()
        self.fps_counter = 0
        
    def start(self) -> bool:
        """
        Inicia la captura de video desde la webcam.
        
        Returns:
            True si la captura inició correctamente, False en caso contrario
        """
        if self.is_capturing:
            print("La captura ya está en ejecución")
            return True
            
        # Inicializar captura de video
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {self.camera_id}")
            return False
        
        # Configurar resolución y FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verificar configuración actual
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Cámara configurada: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        # Iniciar thread de captura
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        print("Captura de webcam iniciada")
        return True
    
    def _capture_loop(self):
        """
        Loop principal de captura (ejecutado en thread separado).
        """
        frame_interval = 1.0 / self.fps
        
        while self.is_capturing:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error al capturar frame")
                time.sleep(0.1)
                continue
            
            # Agregar frame al buffer
            with self.lock:
                self.frame_buffer.append({
                    'frame': frame.copy(),
                    'timestamp': time.time(),
                    'frame_id': self.frames_captured
                })
                self.frames_captured += 1
            
            # Actualizar FPS
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_update >= 1.0:
                self.fps_actual = self.fps_counter / (current_time - self.last_fps_update)
                self.fps_counter = 0
                self.last_fps_update = current_time
            
            # Control de framerate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Obtiene el frame más reciente del buffer.
        
        Returns:
            Frame más reciente o None si el buffer está vacío
        """
        with self.lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer[-1]['frame'].copy()
            return None
    
    def get_frame_batch(self, num_frames: int) -> list:
        """
        Obtiene un lote de frames más recientes.
        
        Args:
            num_frames: Número de frames a obtener
            
        Returns:
            Lista de frames (puede ser menor a num_frames si no hay suficientes)
        """
        with self.lock:
            available = len(self.frame_buffer)
            count = min(num_frames, available)
            
            if count == 0:
                return []
            
            # Obtener los últimos count frames
            frames = [self.frame_buffer[-(count - i)]['frame'].copy() 
                     for i in range(count)]
            return frames
    
    def get_buffer_info(self) -> dict:
        """
        Obtiene información sobre el estado del buffer.
        
        Returns:
            Diccionario con información del buffer
        """
        with self.lock:
            return {
                'buffer_size': len(self.frame_buffer),
                'max_buffer_size': self.buffer_size,
                'frames_captured': self.frames_captured,
                'fps': self.fps_actual,
                'is_capturing': self.is_capturing
            }
    
    def clear_buffer(self):
        """
        Limpia el buffer de frames.
        """
        with self.lock:
            self.frame_buffer.clear()
    
    def stop(self):
        """
        Detiene la captura de video.
        """
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        
        # Esperar a que termine el thread
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
        
        # Liberar recursos
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        print("Captura de webcam detenida")
    
    def __del__(self):
        """
        Destructor: asegura que los recursos se liberen.
        """
        self.stop()


class FramePreprocessor:
    """
    Preprocesa frames para el modelo Wan2.2.
    """
    
    @staticmethod
    def prepare_frame(frame: np.ndarray, target_size: Tuple[int, int] = (720, 1280)) -> np.ndarray:
        """
        Prepara un frame para procesamiento.
        
        Args:
            frame: Frame de entrada (BGR)
            target_size: Tamaño objetivo (height, width)
            
        Returns:
            Frame procesado
        """
        # Redimensionar manteniendo aspect ratio
        h, w = frame.shape[:2]
        target_h, target_w = target_size
        
        # Calcular escala
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Redimensionar
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Agregar padding si es necesario
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        return padded
    
    @staticmethod
    def normalize_frame(frame: np.ndarray) -> np.ndarray:
        """
        Normaliza un frame para el modelo.
        
        Args:
            frame: Frame en formato uint8 [0, 255]
            
        Returns:
            Frame normalizado en [-1, 1]
        """
        return (frame.astype(np.float32) / 127.5) - 1.0


if __name__ == "__main__":
    # Test del capturador
    print("Iniciando test de captura de webcam...")
    
    capture = WebcamCapture(
        camera_id=0,
        width=1280,
        height=720,
        fps=30,
        buffer_size=30
    )
    
    if not capture.start():
        print("Error al iniciar captura")
        exit(1)
    
    print("Presiona 'q' para salir")
    
    try:
        while True:
            frame = capture.get_latest_frame()
            
            if frame is not None:
                # Mostrar información
                info = capture.get_buffer_info()
                cv2.putText(
                    frame,
                    f"FPS: {info['fps']:.1f} | Buffer: {info['buffer_size']}/{info['max_buffer_size']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow('Webcam Capture Test', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.01)
    
    finally:
        capture.stop()
        cv2.destroyAllWindows()
        print("Test finalizado")