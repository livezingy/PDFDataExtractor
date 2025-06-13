# core/detection/table_detector.py
from typing import List, Dict, Optional
from PIL import Image
import time
from dataclasses import dataclass
from core.utils.logger import AppLogger
from core.models.table_models import TableModels

@dataclass
class TableRegion:
    """Table region data class"""
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    page_num: int
    table_type: str

class TableDetector:
    """Optimized table detector with model dependency injection"""
    
    def __init__(self, models: TableModels, config: Optional[Dict] = None):
        """Initialize table detector
        
        Args:
            models: TableModels instance
            config: Optional configuration dictionary
        """
        self.models = models
        self.logger = AppLogger.get_logger()
        
        # Configuration with defaults
        self.config = {
            'min_confidence': 0.7,
            'max_tables_per_page': 10,
            'expand_ratio': 0.1,
            'enable_postprocessing': True
        }
        if config:
            self.config.update(config)

    def detect_tables(
        self, 
        image: Image.Image,
        page_num: int = 0,
        detection_threshold: float = None
    ) -> List[TableRegion]:
        """Optimized detection using shared models
        detection_threshold: user-set threshold from params_panel, fallback to config['min_confidence'] if None
        """
        start_time = time.time()
        try:
            # Use user-set detection_threshold if provided, else fallback to config
            min_conf = detection_threshold if detection_threshold is not None else self.config['min_confidence']
            # Ensure models.detect_tables returns (boxes, scores, labels) or (boxes, scores)
            result = self.models.detect_tables(
                image,
                min_conf
            )
            if len(result) == 3:
                boxes, scores, labels = result
            else:
                boxes, scores = result
            
            # Convert to table regions
            regions = []
            for box, score in zip(boxes, scores):
                region = TableRegion(
                    bbox=tuple(box.tolist()),
                    confidence=float(score),
                    page_num=page_num,
                    table_type=self._determine_table_type(box, image)
                )
                
                if self.config['enable_postprocessing']:
                    region = self._postprocess_region(region, image)
                    
                regions.append(region)

            self.logger.debug(f"detected {len(regions)} table with Transformer in  page {page_num}:")
            
            # Sort and limit results
            regions.sort(key=lambda x: x.confidence, reverse=True)
            return regions[:self.config['max_tables_per_page']]
            
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}", exc_info=True)
            return []
            
    def _determine_table_type(self, box, image: Image.Image) -> str:
        """Table type classification logic"""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        return 'table' if 0.5 <= (width/height) <= 2.0 else 'table_transformed'
            
    def _postprocess_region(self, region: TableRegion, image: Image.Image) -> TableRegion:
        """Region optimization logic"""
        x1, y1, x2, y2 = region.bbox
        expand_x = (x2 - x1) * self.config['expand_ratio']
        expand_y = (y2 - y1) * self.config['expand_ratio']
        
        new_bbox = (
            max(0, x1 - expand_x),
            max(0, y1 - expand_y),
            min(image.width, x2 + expand_x),
            min(image.height, y2 + expand_y)
        )
        return TableRegion(
            bbox=new_bbox,
            confidence=region.confidence,
            page_num=region.page_num,
            table_type=region.table_type
        )