# core/detection/table_extractor.py
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np
from dataclasses import dataclass
import time
from core.models.table_models import TableModels
from .table_structure import TableStructure, Cell
import pandas as pd
from core.utils.logger import AppLogger
from core.processing.ocr_processor import OCRProcessor

@dataclass
class ExtractedTable:
    """Extracted table data class"""
    data: pd.DataFrame
    structure: TableStructure
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = None

class TableExtractor:
    """Table content extractor class
    
    Responsible for extracting table content from images, including:
    1. Text recognition
    2. Data structuring
    3. Table validation
    4. Data export
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize table extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            'min_confidence': 0.5,
            'enable_text_recognition': True,
            'enable_data_validation': True,
            'enable_export': True,
            'export_format': 'excel',
            'export_path': 'output',
            'text_recognition_config': {
                'lang': 'eng',
                'config': '--psm 6'
            }
        }
        if config:
            self.config.update(config)
            
        self.models = TableModels(config)
        self.ocr_processor = OCRProcessor(self.config.get('text_recognition_config'))
        self.logger = AppLogger.get_logger()
        
    def extract_table(
        self,
        image: Image.Image,
        structure: TableStructure,
        table_region: Optional[Tuple[float, float, float, float]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> ExtractedTable:
        """Extract table content
        
        Args:
            image: Input image
            structure: Table structure
            table_region: Table region (x1, y1, x2, y2)
            params: Processing parameters from params_panel
            
        Returns:
            Extracted table data
        """
        start_time = time.time()
        try:
            # Set parameters for structure recognizer
            if params:
                self.models.structure_recognizer.set_params(params)
                
            # If table region specified, crop image
            if table_region:
                image = image.crop(table_region)
                
            # Extract cell texts
            cell_texts = self._extract_cell_texts(image, structure.cells)
            
            # Create data matrix
            data_matrix = self._create_data_matrix(structure, cell_texts)
            
            # Create DataFrame
            df = self._create_dataframe(data_matrix, structure)
            
            # Validate data
            if self.config['enable_data_validation']:
                df = self._validate_data(df)
                
            # Calculate overall confidence
            confidence = self._calculate_confidence(structure, cell_texts)
            
            # Create extraction result
            extracted_table = ExtractedTable(
                data=df,
                structure=structure,
                confidence=confidence,
                processing_time=time.time() - start_time,
                metadata={
                    'image_size': image.size,
                    'table_region': table_region,
                    'cell_count': len(structure.cells)
                }
            )
            
            # Export data if params provided
            if params and self.config['enable_export']:
                self._export_table(extracted_table, params)
                
            self.logger.log_operation("Table extraction", {
                "cell_count": len(structure.cells),
                "confidence": f"{confidence:.2f}",
                "processing_time": f"{extracted_table.processing_time:.2f}s"
            })
            
            return extracted_table
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Table extraction"
            })
            raise
            
    def _extract_cell_texts(
        self,
        image: Image.Image,
        cells: List[Cell]
    ) -> Dict[Tuple[int, int], str]:
        """Extract cell texts
        
        Args:
            image: Input image
            cells: List of cells
            
        Returns:
            Dictionary of cell texts, keyed by (row, col) tuple
        """
        cell_texts = {}
        for cell in cells:
            try:
                # Crop cell region
                x1, y1, x2, y2 = cell.bbox
                cell_image = image.crop((x1, y1, x2, y2))
                
                # Use OCR processor to recognize text
                results = self.ocr_processor.recognize_text(cell_image)
                if results:
                    # Use result with highest confidence
                    best_result = max(results, key=lambda x: x.confidence)
                    text = best_result.text.strip()
                else:
                    text = ""
                
                # Store result
                cell_texts[(cell.row, cell.col)] = text
                
            except Exception as e:
                self.logger.log_exception(e, {
                    "operation": "Cell text extraction",
                    "cell_position": f"({cell.row}, {cell.col})"
                })
                cell_texts[(cell.row, cell.col)] = ""
                
        return cell_texts
        
    def _create_data_matrix(
        self,
        structure: TableStructure,
        cell_texts: Dict[Tuple[int, int], str]
    ) -> List[List[str]]:
        """Create data matrix
        
        Args:
            structure: Table structure
            cell_texts: Dictionary of cell texts
            
        Returns:
            Data matrix
        """
        # Create empty matrix
        matrix = [['' for _ in range(structure.cols)] for _ in range(structure.rows)]
        
        # Fill data
        for cell in structure.cells:
            text = cell_texts.get((cell.row, cell.col), '')
            
            # Handle merged cells
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(cell.col, cell.col + cell.colspan):
                    if 0 <= r < structure.rows and 0 <= c < structure.cols:
                        matrix[r][c] = text
                        
        return matrix
        
    def _create_dataframe(
        self,
        data_matrix: List[List[str]],
        structure: TableStructure
    ) -> pd.DataFrame:
        """Create DataFrame
        
        Args:
            data_matrix: Data matrix
            structure: Table structure
            
        Returns:
            DataFrame object
        """
        # Create DataFrame
        df = pd.DataFrame(data_matrix)
        
        # Set column names
        if structure.header_rows > 0:
            # Use first row as column names
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            
        return df
        
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        try:
            # 1. Clean empty rows
            df = df.dropna(how='all')
            
            # 2. Clean empty columns
            df = df.dropna(axis=1, how='all')
            
            # 3. Clean duplicate rows
            df = df.drop_duplicates()
            
            # 4. Clean duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            
            # 5. Reset index
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Data validation"
            })
            return df
            
    def _calculate_confidence(
        self,
        structure: TableStructure,
        cell_texts: Dict[Tuple[int, int], str]
    ) -> float:
        """Calculate overall confidence
        
        Args:
            structure: Table structure
            cell_texts: Dictionary of cell texts
            
        Returns:
            Confidence score
        """
        try:
            # 1. Structure confidence
            structure_confidence = sum(cell.confidence for cell in structure.cells) / len(structure.cells)
            
            # 2. Text recognition confidence
            text_confidence = len([t for t in cell_texts.values() if t.strip()]) / len(cell_texts)
            
            # 3. Merged cell ratio
            merged_ratio = len([c for c in structure.cells if c.cell_type == 'merged']) / len(structure.cells)
            
            # 4. Calculate combined confidence
            confidence = (
                0.4 * structure_confidence +
                0.4 * text_confidence +
                0.2 * (1 - merged_ratio)  # More merged cells means lower confidence
            )
            
            return min(max(confidence, 0), 1)  # Ensure in [0,1] range
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Confidence calculation"
            })
            return 0.5  # Return default value
            
    def _export_table(self, table: ExtractedTable, params: Dict[str, Any]):
        """Export table data
        
        Args:
            table: Extracted table data
            params: Processing parameters from params_panel
        """
        try:
            # Convert ExtractedTable to dictionary format
            table_dict = {
                'data': table.data.to_dict('records'),
                'columns': table.data.columns.tolist(),
                'confidence': table.confidence,
                'metadata': table.metadata
            }
            
            # Get export parameters from params_panel
            export_format = params.get('export_format', 'csv')
            output_path = params.get('output_path', '')
            
            if not output_path:
                raise ValueError("Output path must be specified")
            
            # Create export parameters
            export_params = {
                'export_format': export_format,
                'output_path': output_path
            }
            
            # Use TextProcessor's export function
            from core.processing.text_processor import TextProcessor
            text_processor = TextProcessor()
            text_processor._handle_export({'tables': [table_dict]}, export_params)
            
            self.logger.log_operation("Table export", {
                "format": export_format,
                "path": output_path
            })
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Table export"
            })
            raise 