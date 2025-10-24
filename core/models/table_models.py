import os
import torch
from typing import Dict, Optional, Tuple, List
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection
)
from PIL import Image
import numpy as np
import pytesseract
import easyocr
from core.utils.easyocr_config import get_easyocr_reader
from tqdm.auto import tqdm
from core.utils.logger import AppLogger
from core.utils.path_utils import get_app_dir

#preprocessing for transformer detection and structure recognition
class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def prepare_image(image, device):
    pixel_values = detection_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    return pixel_values

def prepare_cropped_image(cropped_image, device):
    pixel_values = structure_transform(cropped_image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    return pixel_values
    

class TableModels:
    """Unified model manager for table detection, structure recognition, and OCR."""
    _instance = None
    _initialized = False

    def __new__(cls, config=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_cfg: Optional[Dict] = None):
        self.logger = AppLogger.get_logger()
        
        self.detection_model_path = model_cfg.get('detection_model_path')
        self.structure_model_path = model_cfg.get('structure_model_path')
        self.ocr_model_path = model_cfg.get('ocr_model_path')
        self.device = model_cfg.get('device', 'cpu')
        self.detection_confidence = model_cfg.get('detection_confidence', 0.5)
        self.structure_confidence = model_cfg.get('structure_confidence', 0.5)
        self.ocr_confidence = model_cfg.get('ocr_confidence', 0.5)
        # 调试信息：打印模型路径和参数
        self.logger.debug(f"[TableModels] detection_model_path: {self.detection_model_path}")
        self.logger.debug(f"[TableModels] structure_model_path: {self.structure_model_path}")
        self.logger.debug(f"[TableModels] ocr_model_path: {self.ocr_model_path}")
        self.logger.debug(f"[TableModels] device: {self.device}")
        self.logger.debug(f"[TableModels] detection_confidence: {self.detection_confidence}")
        self.logger.debug(f"[TableModels] structure_confidence: {self.structure_confidence}")
        self.logger.debug(f"[TableModels] ocr_confidence: {self.ocr_confidence}")

        self.models = {}
        self.processors = {}
        self._init()
        self._initialized = True
        
        pytesseract.pytesseract.tesseract_cmd = self.ocr_model_path

    def _init(self):
        try:
            # Detection model - 强制使用本地模型，禁止自动下载
            self.models['detection'] = TableTransformerForObjectDetection.from_pretrained(
                self.detection_model_path,
                local_files_only=True
            ).to(self.device)
            self.processors['detection'] = AutoImageProcessor.from_pretrained(
                self.detection_model_path,
                local_files_only=True
            )
            # Structure model - 强制使用本地模型，禁止自动下载
            self.models['structure'] = TableTransformerForObjectDetection.from_pretrained(
                self.structure_model_path,
                local_files_only=True
            ).to(self.device)
            self.processors['structure'] = AutoImageProcessor.from_pretrained(
                self.structure_model_path,
                local_files_only=True
            )
            # Set eval mode
            for model in self.models.values():
                model.eval()
            self.logger.info("Models initialized successfully", {
                "models": list(self.models.keys()),
                "device": self.device
            })
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}", exc_info=True)
            raise

    def detect_tables(self, image: Image.Image):
        """Detect tables in image, return boxes, scores, labels."""
        try:
            inputs = self.processors['detection'](
                images=image,
                return_tensors="pt",
                size={"shortest_edge": 1024, "longest_edge": 1024}
            )
            with torch.no_grad():
                outputs = self.models['detection'](**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processors['detection'].post_process_object_detection(
                outputs,
                threshold=self.detection_confidence,
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

    

    def recognize_structure(self, image: Image.Image):
        """
        Recognize table structure using direct detection method (similar to table_parser_direct.py).
        Return tuple: (model, outputs, image_size)
        - model: structure detection model
        - outputs: raw model outputs (not post-processed)
        - image_size: original image size for coordinate scaling
        """
        try:
            processor = self.processors['structure']
            model = self.models['structure']
            
            # Use the same preprocessing as table_parser_direct.py
            inputs = processor(
                images=image,
                return_tensors="pt",
                size={"shortest_edge": 1000, "longest_edge": 1000}
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Return raw outputs instead of post-processed results
            # This allows table_parser.py to use the same coordinate processing as table_parser_direct.py
            return model, outputs, image.size
            
        except Exception as e:
            self.logger.error(f"Structure recognition failed: {str(e)}", exc_info=True)
            raise

    

    def ocr_cell(self, image: Image.Image, lang: str = 'eng') -> Tuple[str, float]:
        """
        OCR for a single cell image using Tesseract.
        Returns tuple: (text, confidence_score)
        """
        try:
            ocr_data = pytesseract.image_to_data(
            image, lang=lang, config='--psm 6 preserve_interword_spaces', output_type=pytesseract.Output.DICT
            )
            # 以单词为单位聚合
            words = []
            confidences = []
            for i, word in enumerate(ocr_data['text']):
                conf = ocr_data['conf'][i]
                if word.strip() and conf > -1:
                    words.append(word)
                    confidences.append(conf)
            text = ' '.join(words).strip()
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
                    
            # Apply confidence threshold
            if avg_confidence < self.ocr_confidence:
                return "", avg_confidence
                
            return text, avg_confidence
            
        except Exception as e:
            self.logger.error(f"OCR failed: {str(e)}", exc_info=True)
            return "", 0.0



