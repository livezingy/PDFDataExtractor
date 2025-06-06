import os
import torch
from typing import Dict, Optional, Tuple
from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection,
    TrOCRProcessor,
    VisionEncoderDecoderModel
)
from dataclasses import dataclass
import time
from core.utils.logger import AppLogger
import sys
from PIL import Image
from typing import Dict, List, Optional, Tuple
import numpy as np
from core.utils.path_utils import get_app_dir

@dataclass
class ModelConfig:
    """Model configuration data class"""
    model_path: str
    device: str
    confidence_threshold: float
    max_image_size: int
    batch_size: int

class TableModels:
    """Table model management class
    
    Responsible for loading and managing various table processing models, including:
    1. Table detection model (Table Transformer)
    2. Table structure recognition model (Table Structure Recognition)
    3. Text recognition model (TrOCR)
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, config=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize model manager
        
        Args:
            config: Model configuration dictionary
        """
        # Skip initialization if already done
        if self._initialized:
            return
            
        # Initialize logger first
        self.logger = AppLogger.get_logger()
        base_dir = get_app_dir()        
            
        self.config = {
            'detection_model_path': os.path.join(base_dir, 'models', 'table-transformer', 'detection'),
            'structure_model_path': os.path.join(base_dir, 'models', 'table-transformer', 'structure'),
            'ocr_model_path': os.path.join(base_dir, 'models', 'table-transformer', 'ocr'),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'confidence_threshold': 0.7,
            'max_image_size': 2048,
            'batch_size': 1
        }
        if config:
            self.config.update(config)
            
        # Initialize model containers
        self.models = {}
        self.processors = {}
        
        # Initialize models
        self._init()
        
        # Mark as initialized
        self._initialized = True
        
    def _init(self, config: Optional[Dict] = None):
        """Initialize models and processors
        
        Args:
            config: Optional configuration dictionary
        """
        try:
            # Update config if provided
            if config:
                self.config.update(config)
                
            # Load table detection model
            self.models['detection'] = TableTransformerForObjectDetection.from_pretrained(
                self.config['detection_model_path']
            ).to(self.config['device'])
            self.processors['detection'] = AutoImageProcessor.from_pretrained(
                self.config['detection_model_path']
            )
            
            # Load table structure recognition model
            self.models['structure'] = TableTransformerForObjectDetection.from_pretrained(
                self.config['structure_model_path']
            ).to(self.config['device'])
            self.processors['structure'] = AutoImageProcessor.from_pretrained(
                self.config['structure_model_path']
            )
            
            # Set models to evaluation mode
            for model in self.models.values():
                model.eval()
                
            self.logger.info("Models initialized successfully", {
                "models": list(self.models.keys()),
                "device": self.config['device']
            })
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}", exc_info=True)
            raise
        
    def _load_models(self):
        """Load all models"""
        try:
            # Load table detection model
            self.models['detection'] = TableTransformerForObjectDetection.from_pretrained(
                self.config['detection_model_path']
            ).to(self.config['device'])
            self.processors['detection'] = AutoImageProcessor.from_pretrained(
                self.config['detection_model_path']
            )
            
            # Load table structure recognition model
            self.models['structure'] = TableTransformerForObjectDetection.from_pretrained(
                self.config['structure_model_path']
            ).to(self.config['device'])
            self.processors['structure'] = AutoImageProcessor.from_pretrained(
                self.config['structure_model_path']
            )
            
            """    # Load OCR model
            self.models['ocr'] = VisionEncoderDecoderModel.from_pretrained(
                self.config['ocr_model_path']
            ).to(self.config['device'])
            self.processors['ocr'] = TrOCRProcessor.from_pretrained(
                self.config['ocr_model_path']
            ) """
            
            # Set models to evaluation mode
            for model in self.models.values():
                model.eval()
                
            self.logger.log_operation("Model loading", {
                "status": "Success",
                "models": list(self.models.keys())
            })
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Model loading"
            })
            raise
            
    def get_model_info(self) -> Dict:
        """Get model information
        
        Returns:
            Dictionary containing model configuration and status information
        """
        return {
            'detection_model': {
                'path': self.config['detection_model_path'],
                'loaded': 'detection' in self.models,
                'device': self.config['device']
            },
            'structure_model': {
                'path': self.config['structure_model_path'],
                'loaded': 'structure' in self.models,
                'device': self.config['device']
            },
            'ocr_model': {
                'path': self.config['ocr_model_path'],
                'loaded': 'ocr' in self.models,
                'device': self.config['device']
            }
        }
        
    def update_config(self, new_config: Dict):
        """Update model configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.logger.log_operation("Configuration update", {
            "new_config": new_config
        })

    def detect_tables(
        self,
        image: Image.Image,
        min_confidence: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect tables in image
        
        Args:
            image: Input image
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (boxes, scores, labels)
            - boxes: Array of bounding boxes (N, 4)
            - scores: Array of confidence scores (N,)
            - labels: Array of labels (N,)
        """
        try:
            # Process image
            inputs = self.processors['detection'](
                images=image,
                return_tensors="pt",
                size={"shortest_edge": 1024, "longest_edge": 1024}
            )
            
            # Run inference
            with torch.no_grad():
                outputs = self.models['detection'](**inputs)
                
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processors['detection'].post_process_object_detection(
                outputs,
                threshold=min_confidence,
                target_sizes=target_sizes
            )[0]
            
            return (
                results["boxes"].cpu().numpy(),
                results["scores"].cpu().numpy(),
                results["labels"].cpu().numpy()
            )
            
        except Exception as e:
            self.logger.error(f"Table detection failed: {str(e)}", exc_info=True)
            raise
            
    def recognize_structure(
        self,
        image: Image.Image,
        min_confidence: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recognize table structure
        
        Args:
            image: Input image
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (boxes, scores)
            - boxes: Array of bounding boxes (N, 4)
            - scores: Array of confidence scores (N,)
        """
        try:
            # Process image
            inputs = self.processors['structure'](
                images=image,
                return_tensors="pt",
                size={"shortest_edge": 1024, "longest_edge": 1024}
            )
            
            # Run inference
            with torch.no_grad():
                outputs = self.models['structure'](**inputs)
                
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processors['structure'].post_process_object_detection(
                outputs,
                threshold=min_confidence,
                target_sizes=target_sizes
            )[0]
            
            return (
                results["boxes"].cpu().numpy(),
                results["scores"].cpu().numpy()
            )
            
        except Exception as e:
            self.logger.error(f"Structure recognition failed: {str(e)}", exc_info=True)
            raise
        
        
    