# core/processing/table_processor.py
from typing import Dict, List, Optional, Any
from PIL import Image
from dataclasses import dataclass
import time
from core.detection.table_detector import TableDetector, TableRegion
from core.detection.table_structure import TableStructure, TableStructureRecognizer
from core.detection.table_extractor import TableExtractor, ExtractedTable
from core.utils.logger import AppLogger
from core.models.table_models import TableModels

@dataclass
class ProcessingResult:
    tables: List[ExtractedTable]
    total_time: float
    success: bool
    error: Optional[str] = None

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
        self.structure_recognizer = TableStructureRecognizer(self.models, config)
        self.extractor = TableExtractor(config)
        
        # Configuration
        self.config = {
            'min_confidence': 0.5,
            'export_format': 'excel',
            'enable_validation': True
        }
        if config:
            self.config.update(config)

    def detect_tables(self, image: Image.Image, params: Dict) -> List[Dict]:
        """Unified detection interface"""
        try:
            raw_regions = self.detector.detect_tables(
                image,
                page_num=params.get('page_num', 1)
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
        """Optimized region processing pipeline"""
        try:
            # Crop and process
            x1, y1, x2, y2 = region['bbox']
            cropped = image.crop((x1, y1, x2, y2))
            
            # Structure recognition
            structure = self.structure_recognizer.recognize_structure(cropped,params.get('current_filepath', ''))
            
            # Content extraction
            table = self.extractor.extract_table(
                cropped,
                structure,
                region['bbox']
            )
            
            if table.confidence >= self.config['min_confidence']:
                return {
                    'data': table.data.to_dict('records'),
                    'columns': table.data.columns.tolist(),
                    'confidence': table.confidence,
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

    
                
   
            
            
            
    