# core/processing/ocr_processor.py
from sklearn import base
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from core.utils.logger import AppLogger
from core.utils.path_utils import get_app_dir
from sympy import im
import os

@dataclass
class OCRResult:
    """OCR recognition result data class"""
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    page_num: int
    processing_time: float

class OCRProcessor:
    """OCR processor class
    
    Provides unified OCR processing interface, supporting multiple OCR engines and configuration options.
    Currently mainly supports Tesseract OCR.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize OCR processor
        
        Args:
            config: OCR configuration dictionary, containing optional parameters:
                - tesseract_cmd: Tesseract executable path
                - lang: Recognition language
                - config: Tesseract configuration parameters
                - preprocessing: Whether to enable image preprocessing
        """
        base_dir = get_app_dir()
        self.config = {
            'tesseract_cmd': os.path.join(base_dir, 'models', 'Tesseract-OCR', 'tesseract.exe') if os.name == 'nt' else 'tesseract',
            'tessdata_dir': os.path.join(base_dir, 'models', 'Tesseract-OCR', 'tessdata') if os.name == 'nt' else '/usr/share/tesseract-ocr/4.00/tessdata',
            'lang': 'eng',
            'config': '--psm 6',
            'preprocessing': True,
            'confidence_threshold': 60
        }
        if config:
            self.config.update(config)
            
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
        self.logger = AppLogger.get_logger()
        
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR recognition rate
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        if not self.config['preprocessing']:
            return image
            
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
                
            # Convert to numpy array
            img_array = np.array(image)
            
            # Binarization
            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Denoising
            denoised = cv2.fastNlMeansDenoising(binary, None, 5, 7, 21)
            
            return Image.fromarray(denoised)
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Image preprocessing"
            })
            return image
        
    def recognize_text(
        self, 
        image: Image.Image,
        page_num: int = 0,
        region: Optional[Tuple[float, float, float, float]] = None
    ) -> List[OCRResult]:
        """Recognize text in image
        
        Args:
            image: Input image
            page_num: Page number
            region: Recognition region (x1, y1, x2, y2), if None process entire image
            
        Returns:
            List of OCR recognition results
        """
        start_time = time.time()
        
        try:
            # If region specified, crop image
            if region:
                image = image.crop(region)
                
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Execute OCR
            data = pytesseract.image_to_data(
                processed_image,
                lang=self.config['lang'],
                config=self.config['config'],
                output_type=pytesseract.Output.DICT
            )
            
            # Process results
            results = []
            for i in range(len(data['text'])):
                if float(data['conf'][i]) > self.config['confidence_threshold'] and data['text'][i].strip():
                    # Calculate coordinates
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    # If region specified, adjust coordinates
                    if region:
                        x += region[0]
                        y += region[1]
                        
                    results.append(OCRResult(
                        text=data['text'][i],
                        confidence=float(data['conf'][i]),
                        bbox=(x, y, x + w, y + h),
                        page_num=page_num,
                        processing_time=time.time() - start_time
                    ))                   
            
                    
            return results
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "[Tesseract] Text recognition"
            })
            raise
            
    def recognize_table_cell(
        self,
        cell_image: Image.Image,
        cell_bbox: Tuple[float, float, float, float],
        page_num: int = 0
    ) -> Optional[OCRResult]:
        """Recognize text in table cell
        
        Args:
            cell_image: Cell image
            cell_bbox: Cell bounding box (x1, y1, x2, y2)
            page_num: Page number
            
        Returns:
            OCR recognition result, returns None if recognition fails
        """
        try:
            results = self.recognize_text(cell_image, page_num)
            if results:
                # Use result with highest confidence
                best_result = max(results, key=lambda x: x.confidence)
                # Update bounding box to cell bounding box
                best_result.bbox = cell_bbox
                return best_result
            return None
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Cell text recognition",
                "cell_bbox": cell_bbox
            })
            return None 