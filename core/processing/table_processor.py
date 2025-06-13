# core/processing/table_processor.py
from typing import Dict, List, Optional, Any
from PIL import Image
import time
from core.detection.table_detector import TableDetector, TableRegion
from core.utils.logger import AppLogger
from core.models.table_models import TableModels
from core.detection.table_parser import TableParser


class TableProcessor:
    """Optimized processor with dependency injection"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize table processor
        
        Args:
            config: Optional configuration dictionary
        """
        # Initialize models first
        self.models = TableModels(config)
        self.logger = AppLogger.get_logger()
        
        # Initialize components with shared models
        self.detector = TableDetector(self.models, config)
        self.table_parser = TableParser(config)
        
        # Configuration
        self.config = {
            'min_confidence': 0.3,
            'export_format': 'excel',
            'enable_validation': True,
            'detection_threshold': 0.5,
            'structure_threshold': 0.5,
            'structure_border_width': 5,
            'structure_preprocess': True
        }
        if config:
            self.config.update(config)

    def detect_tables(self, image: Image.Image, params: Dict) -> List[Dict]:
        """Unified detection interface"""
        try:
            # Use detection threshold from params if present
            min_conf = params.get('detection_threshold', self.config.get('detection_threshold', 0.5))
            raw_regions = self.detector.detect_tables(
                image,
                page_num=params.get('page_num', 1),
                detection_threshold=min_conf
            )
            return [self._region_to_dict(r) for r in raw_regions]
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            return []

    def process_region(
        self,
        image: Image.Image,
        region: Dict,
        params: Dict
    ) -> Optional[Dict]:
        """Optimized region processing pipeline using TableParser"""
        try:
            # 1. Crop image to region bbox
            x1, y1, x2, y2 = region['bbox']
            cropped_image = image.crop((x1, y1, x2, y2))
            # 2. Use TableParser to process the cropped table image
            # Pass structure params
            parse_params = {
                'structure_threshold': params.get('structure_threshold', self.config.get('structure_threshold', 0.5)),
                'structure_border_width': params.get('structure_border_width', self.config.get('structure_border_width', 5)),
                'structure_preprocess': params.get('structure_preprocess', self.config.get('structure_preprocess', True)),
                'min_confidence': params.get('detection_threshold', self.config.get('detection_threshold', 0.5))                
            }
            parsed_table = self.table_parser.parse_table(cropped_image, region['bbox'], parse_params)
            if parsed_table and parsed_table.get('confidence', 0) >= self.config['min_confidence']:
                return {
                    'data': parsed_table['data'],
                    'columns': parsed_table['columns'],
                    'confidence': parsed_table['confidence'],
                    'bbox': region['bbox'],
                    'page': params.get('page_num', 1)
                }
            return None
        except Exception as e:
            self.logger.error(f"Region processing failed: {str(e)}")
            return None

    def _region_to_dict(self, region: TableRegion) -> Dict:
        """Convert region object to dictionary"""
        return {
            'bbox': region.bbox,
            'confidence': region.confidence,
            'page': region.page_num,
            'type': region.table_type
        }







