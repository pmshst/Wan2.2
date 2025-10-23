# Copyright 2024-2025 Real-time Face Swap Extension for Wan2.2
"""
Tests unitarios para el sistema de Real-Time Face Swap.
"""

import os
import sys
import unittest
import numpy as np
import cv2
import time
from pathlib import Path

# Agregar directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_face_swap.webcam_capture import WebcamCapture, FramePreprocessor


class TestWebcamCapture(unittest.TestCase):
    """Tests para WebcamCapture."""
    
    def setUp(self):
        """Configuración antes de cada test."""
        self.capture = None
    
    def tearDown(self):
        """Limpieza después de cada test."""
        if self.capture is not None:
            self.capture.stop()
    
    def test_init(self):
        """Test de inicialización."""
        self.capture = WebcamCapture(
            camera_id=0,
            width=640,
            height=480,
            fps=30,
            buffer_size=10
        )
        
        self.assertEqual(self.capture.camera_id, 0)
        self.assertEqual(self.capture.width, 640)
        self.assertEqual(self.capture.height, 480)
        self.assertEqual(self.capture.fps, 30)
        self.assertEqual(self.capture.buffer_size, 10)
    
    def test_buffer_info(self):
        """Test de información del buffer."""
        self.capture = WebcamCapture(buffer_size=5)
        
        info = self.capture.get_buffer_info()
        
        self.assertIn('buffer_size', info)
        self.assertIn('max_buffer_size', info)
        self.assertIn('frames_captured', info)
        self.assertIn('fps', info)
        self.assertIn('is_capturing', info)
        
        self.assertEqual(info['max_buffer_size'], 5)
        self.assertFalse(info['is_capturing'])


class TestFramePreprocessor(unittest.TestCase):
    """Tests para FramePreprocessor."""
    
    def test_prepare_frame_resize(self):
        """Test de redimensionamiento de frame."""
        # Crear frame de prueba
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Preparar frame
        prepared = FramePreprocessor.prepare_frame(frame, target_size=(720, 1280))
        
        # Verificar tamaño
        self.assertEqual(prepared.shape, (720, 1280, 3))
        self.assertEqual(prepared.dtype, np.uint8)
    
    def test_prepare_frame_aspect_ratio(self):
        """Test de mantener aspect ratio."""
        # Frame cuadrado
        frame_square = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        prepared = FramePreprocessor.prepare_frame(frame_square, target_size=(720, 1280))
        
        # Debería tener padding
        self.assertEqual(prepared.shape, (720, 1280, 3))
    
    def test_normalize_frame(self):
        """Test de normalización de frame."""
        # Frame en rango [0, 255]
        frame = np.full((100, 100, 3), 127, dtype=np.uint8)
        
        # Normalizar
        normalized = FramePreprocessor.normalize_frame(frame)
        
        # Verificar rango [-1, 1]
        self.assertAlmostEqual(normalized.mean(), -0.00392157, places=5)
        self.assertTrue(normalized.min() >= -1.0)
        self.assertTrue(normalized.max() <= 1.0)
        self.assertEqual(normalized.dtype, np.float32)


class TestIntegration(unittest.TestCase):
    """Tests de integración básicos."""
    
    def test_module_imports(self):
        """Test de importación de módulos."""
        try:
            from realtime_face_swap import WebcamCapture
            from realtime_face_swap import FramePreprocessor
            from realtime_face_swap import RealtimeFaceSwap
            from realtime_face_swap import AsyncFaceSwapProcessor
            from realtime_face_swap import RealtimeFaceSwapApp
            
            # Si llegamos aquí, todas las importaciones funcionaron
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Error al importar módulos: {e}")
    
    def test_version_info(self):
        """Test de información de versión."""
        import realtime_face_swap
        
        self.assertTrue(hasattr(realtime_face_swap, '__version__'))
        self.assertTrue(hasattr(realtime_face_swap, '__author__'))
        self.assertTrue(hasattr(realtime_face_swap, '__description__'))


class TestFileStructure(unittest.TestCase):
    """Tests de estructura de archivos."""
    
    def test_required_files_exist(self):
        """Test de que existan los archivos requeridos."""
        base_dir = Path(__file__).parent
        
        required_files = [
            'webcam_capture.py',
            'face_swap_processor.py',
            'realtime_app.py',
            'example_simple.py',
            '__init__.py',
            'README.md',
            'QUICKSTART.md',
            'ARCHITECTURE.md',
            'config.ini'
        ]
        
        for filename in required_files:
            file_path = base_dir / filename
            self.assertTrue(
                file_path.exists(),
                f"Archivo requerido no encontrado: {filename}"
            )
    
    def test_documentation_not_empty(self):
        """Test de que la documentación no esté vacía."""
        base_dir = Path(__file__).parent
        
        doc_files = ['README.md', 'QUICKSTART.md', 'ARCHITECTURE.md']
        
        for filename in doc_files:
            file_path = base_dir / filename
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                self.assertGreater(
                    len(content),
                    100,
                    f"Documentación muy corta: {filename}"
                )


def run_tests():
    """Ejecuta todos los tests."""
    # Crear suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar tests
    suite.addTests(loader.loadTestsFromTestCase(TestWebcamCapture))
    suite.addTests(loader.loadTestsFromTestCase(TestFramePreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestFileStructure))
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Retornar código de salida
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    print("="*70)
    print("  Tests Unitarios - Real-Time Face Swap")
    print("="*70)
    print()
    
    exit_code = run_tests()
    
    print()
    print("="*70)
    if exit_code == 0:
        print("  ✓ Todos los tests pasaron correctamente")
    else:
        print("  ✗ Algunos tests fallaron")
    print("="*70)
    
    sys.exit(exit_code)
