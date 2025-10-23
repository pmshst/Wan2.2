# Copyright 2024-2025 Real-time Face Swap Extension for Wan2.2
"""
Procesador de Face Swap en tiempo real optimizado para baja latencia.
Utiliza el modelo Wan2.2-Animate con procesamiento por lotes y caché.
"""

import os
import cv2
import numpy as np
import torch
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import deque
import tempfile
import shutil

# Importar módulos de Wan2.2
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.animate import WanAnimate
from wan.modules.animate.preprocess.process_pipepline import ProcessPipeline
from decord import VideoReader


class RealtimeFaceSwap:
    """
    Procesador de Face Swap en tiempo real usando Wan2.2-Animate.
    """
    
    def __init__(
        self,
        config,
        checkpoint_dir: str,
        preprocess_checkpoint_dir: str,
        reference_image_path: str,
        device_id: int = 0,
        batch_size: int = 8,
        clip_len: int = 25,  # Reducido para menor latencia
        use_gpu: bool = True,
        offload_model: bool = False,
    ):
        """
        Inicializa el procesador de Face Swap.
        
        Args:
            config: Configuración del modelo Wan2.2
            checkpoint_dir: Directorio con los checkpoints del modelo
            preprocess_checkpoint_dir: Directorio con checkpoints de preprocesamiento
            reference_image_path: Ruta a la imagen de referencia para face swap
            device_id: ID del dispositivo GPU
            batch_size: Número de frames a procesar en lote
            clip_len: Longitud del clip (debe ser 4n+1)
            use_gpu: Usar GPU para procesamiento
            offload_model: Descargar modelo a CPU cuando no se usa
        """
        self.checkpoint_dir = checkpoint_dir
        self.preprocess_checkpoint_dir = preprocess_checkpoint_dir
        self.reference_image_path = reference_image_path
        self.device_id = device_id
        self.batch_size = batch_size
        self.clip_len = clip_len
        self.use_gpu = use_gpu
        self.offload_model = offload_model
        
        # Validar clip_len
        if (clip_len - 1) % 4 != 0:
            raise ValueError(f"clip_len debe ser 4n+1, recibido: {clip_len}")
        
        self.device = torch.device(f'cuda:{device_id}' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        print(f"Inicializando RealtimeFaceSwap en {self.device}")
        
        # Inicializar pipeline de preprocesamiento
        print("Cargando pipeline de preprocesamiento...")
        self.preprocess_pipeline = self._init_preprocess_pipeline()
        
        # Inicializar modelo Wan2.2-Animate
        print("Cargando modelo Wan2.2-Animate...")
        self.wan_animate = WanAnimate(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,  # Mantener T5 en CPU para ahorrar VRAM
            convert_model_dtype=False,
            use_relighting_lora=False
        )
        
        # Buffer para frames procesados
        self.processed_buffer = deque(maxlen=60)
        self.buffer_lock = threading.Lock()
        
        # Directorio temporal para procesamiento
        self.temp_dir = tempfile.mkdtemp(prefix="realtime_faceswap_")
        
        # Caché de datos preprocesados de referencia
        self.reference_cache = None
        self.preprocess_reference()
        
        # Estadísticas
        self.processing_fps = 0.0
        self.processing_latency = 0.0
        self.frames_processed = 0
        
        print("RealtimeFaceSwap inicializado correctamente")
    
    def _init_preprocess_pipeline(self) -> ProcessPipeline:
        """
        Inicializa el pipeline de preprocesamiento.
        """
        det_checkpoint = os.path.join(self.preprocess_checkpoint_dir, "det.pth")
        pose2d_checkpoint = os.path.join(self.preprocess_checkpoint_dir, "pose2d.pth")
        sam_checkpoint = os.path.join(self.preprocess_checkpoint_dir, "sam2_hiera_large.pt")
        flux_kontext_path = None  # No usado en tiempo real
        
        return ProcessPipeline(
            det_checkpoint_path=det_checkpoint,
            pose2d_checkpoint_path=pose2d_checkpoint,
            sam_checkpoint_path=sam_checkpoint,
            flux_kontext_path=flux_kontext_path
        )
    
    def preprocess_reference(self):
        """
        Preprocesa la imagen de referencia y la cachea.
        """
        print(f"Preprocesando imagen de referencia: {self.reference_image_path}")
        
        # Cargar imagen de referencia
        if not os.path.exists(self.reference_image_path):
            raise FileNotFoundError(f"Imagen de referencia no encontrada: {self.reference_image_path}")
        
        # Guardar en caché para uso futuro
        self.reference_cache = {
            'path': self.reference_image_path,
            'preprocessed': True
        }
        
        print("Imagen de referencia cacheada")
    
    def preprocess_frames(self, frames: List[np.ndarray], output_dir: str) -> bool:
        """
        Preprocesa un lote de frames para face swap.
        
        Args:
            frames: Lista de frames (BGR, uint8)
            output_dir: Directorio para guardar resultados preprocesados
            
        Returns:
            True si el preprocesamiento fue exitoso
        """
        try:
            # Crear video temporal desde frames
            temp_video_path = os.path.join(self.temp_dir, "temp_input.mp4")
            
            # Guardar frames como video
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
            # Ejecutar preprocesamiento
            success = self.preprocess_pipeline(
                video_path=temp_video_path,
                refer_image_path=self.reference_image_path,
                output_path=output_dir,
                resolution_area=[width, height],
                fps=30,
                iterations=3,
                k=7,
                w_len=1,
                h_len=1,
                retarget_flag=False,
                use_flux=False,
                replace_flag=True  # Modo de reemplazo para face swap
            )
            
            # Limpiar video temporal
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
            return success
            
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return False
    
    def process_frame_batch(self, frames: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        """
        Procesa un lote de frames y aplica face swap.
        
        Args:
            frames: Lista de frames a procesar (BGR, uint8)
            
        Returns:
            Lista de frames procesados o None si hay error
        """
        if len(frames) == 0:
            return None
        
        start_time = time.time()
        
        try:
            # Crear directorio temporal para este lote
            batch_temp_dir = os.path.join(self.temp_dir, f"batch_{self.frames_processed}")
            os.makedirs(batch_temp_dir, exist_ok=True)
            
            # Preprocesar frames
            preprocess_success = self.preprocess_frames(frames, batch_temp_dir)
            
            if not preprocess_success:
                print("Error en preprocesamiento")
                return None
            
            # Generar video con face swap
            with torch.no_grad():
                output_video = self.wan_animate.generate(
                    src_root_path=batch_temp_dir,
                    replace_flag=True,
                    clip_len=min(self.clip_len, len(frames) + 4 - (len(frames) % 4)),
                    refert_num=1,
                    shift=5.0,
                    sample_solver='dpm++',
                    sampling_steps=15,  # Reducido para velocidad
                    guide_scale=1.0,
                    input_prompt="",
                    n_prompt="",
                    seed=42,
                    offload_model=self.offload_model
                )
            
            # Convertir tensor a frames
            if output_video is not None:
                # output_video shape: [C, T, H, W]
                output_frames = []
                
                # Convertir de tensor a numpy
                video_np = output_video.cpu().numpy()
                video_np = np.transpose(video_np, (1, 2, 3, 0))  # [T, H, W, C]
                
                # Desnormalizar y convertir a uint8
                video_np = ((video_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                
                # Convertir RGB a BGR para OpenCV
                for i in range(video_np.shape[0]):
                    frame_bgr = cv2.cvtColor(video_np[i], cv2.COLOR_RGB2BGR)
                    output_frames.append(frame_bgr)
                
                # Limpiar directorio temporal
                shutil.rmtree(batch_temp_dir, ignore_errors=True)
                
                # Actualizar estadísticas
                elapsed = time.time() - start_time
                self.processing_latency = elapsed
                self.processing_fps = len(frames) / elapsed if elapsed > 0 else 0
                self.frames_processed += len(frames)
                
                return output_frames
            
            return None
            
        except Exception as e:
            print(f"Error procesando lote: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_single_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesa un único frame (envuelve process_frame_batch).
        
        Args:
            frame: Frame a procesar
            
        Returns:
            Frame procesado o None
        """
        result = self.process_frame_batch([frame])
        if result and len(result) > 0:
            return result[0]
        return None
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas de procesamiento.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'processing_fps': self.processing_fps,
            'processing_latency': self.processing_latency,
            'frames_processed': self.frames_processed,
            'buffer_size': len(self.processed_buffer),
            'device': str(self.device)
        }
    
    def add_to_buffer(self, frame: np.ndarray):
        """
        Agrega un frame procesado al buffer.
        
        Args:
            frame: Frame procesado
        """
        with self.buffer_lock:
            self.processed_buffer.append(frame)
    
    def get_from_buffer(self) -> Optional[np.ndarray]:
        """
        Obtiene el frame más reciente del buffer.
        
        Returns:
            Frame más reciente o None
        """
        with self.buffer_lock:
            if len(self.processed_buffer) > 0:
                return self.processed_buffer[-1]
            return None
    
    def cleanup(self):
        """
        Limpia recursos temporales.
        """
        print("Limpiando recursos...")
        
        # Limpiar directorio temporal
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        print("Limpieza completada")
    
    def __del__(self):
        """
        Destructor: limpia recursos.
        """
        self.cleanup()


class AsyncFaceSwapProcessor:
    """
    Procesador asíncrono que gestiona el procesamiento de face swap en un thread separado.
    """
    
    def __init__(self, face_swap_processor: RealtimeFaceSwap, batch_size: int = 8):
        """
        Inicializa el procesador asíncrono.
        
        Args:
            face_swap_processor: Instancia de RealtimeFaceSwap
            batch_size: Tamaño del lote para procesamiento
        """
        self.processor = face_swap_processor
        self.batch_size = batch_size
        
        # Cola de entrada
        self.input_queue = deque(maxlen=60)
        self.input_lock = threading.Lock()
        
        # Estado
        self.is_processing = False
        self.processing_thread = None
    
    def start(self):
        """
        Inicia el procesamiento asíncrono.
        """
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        print("Procesamiento asíncrono iniciado")
    
    def _processing_loop(self):
        """
        Loop de procesamiento (ejecutado en thread separado).
        """
        while self.is_processing:
            # Obtener lote de frames
            batch = []
            with self.input_lock:
                while len(self.input_queue) > 0 and len(batch) < self.batch_size:
                    batch.append(self.input_queue.popleft())
            
            if len(batch) > 0:
                # Procesar lote
                processed = self.processor.process_frame_batch(batch)
                
                if processed:
                    # Agregar resultados al buffer
                    for frame in processed:
                        self.processor.add_to_buffer(frame)
            else:
                time.sleep(0.01)  # Esperar si no hay frames
    
    def add_frame(self, frame: np.ndarray):
        """
        Agrega un frame a la cola de procesamiento.
        
        Args:
            frame: Frame a procesar
        """
        with self.input_lock:
            self.input_queue.append(frame)
    
    def stop(self):
        """
        Detiene el procesamiento asíncrono.
        """
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("Procesamiento asíncrono detenido")
