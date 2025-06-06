# core/detection/table_structure.py
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw
import time
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from core.utils.logger import AppLogger
from core.models.table_models import TableModels
import logging
from core.utils.path_utils import get_output_subpath
import os

@dataclass
class Cell:
    bbox: tuple
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    confidence: float = 1.0
    cell_type: str = 'data'

@dataclass
class TableStructure:
    cells: List[Cell]
    rows: int
    cols: int
    header_rows: int = 0

class TableStructureRecognizer:
    """Optimized structure recognizer with model service"""
    
    def __init__(self, models: TableModels, config: Optional[Dict] = None):
        self.models = models
        self.logger = AppLogger.get_logger()
        self.params = config or {}  # Initialize params
        
        if 'output_path' not in self.params or not self.params.get('output_path'):
            self.params['output_path'] = getattr(self.logger, 'config', {}).get('output_path', '')
        self.config = {
            'min_confidence': 0.5,
            'max_cells': 1000,
            'row_cluster_threshold': 0.1,
            'col_cluster_threshold': 0.1
        }
        if config:
            self.config.update(config)
            
    def set_params(self, params: Dict):
        """Set processing parameters
        
        Args:
            params: Processing parameters
        """
        #  output_path
        if 'output_path' not in params or not params['output_path']:
            params['output_path'] = getattr(self.logger, 'config', {}).get('output_path', '')
        self.params = params

    def recognize_structure(
        self,
        image: Image.Image,
        file_path: Optional[str] = None
    ) -> TableStructure:
        """Optimized recognition using model service"""
        # make sure that there is value in self.params['output_path']
        if 'output_path' not in self.params or not self.params['output_path']:
            self.params['output_path'] = getattr(self.logger, 'config', {}).get('output_path', '')
        start_time = time.time()
        try:
            # Use model service for structure recognition
            boxes, scores = self.models.recognize_structure(
                image,
                self.config['min_confidence']
            )
            
            self.logger.debug(f"Structure recognition results:", {
                'total_cells_detected': len(boxes),
                'image_size': image.size,
                'min_confidence': self.config['min_confidence']
            })

            # Convert to cell objects
            cells = []
            for idx, (box, score) in enumerate(zip(boxes, scores)):
                if score >= self.config['min_confidence']:
                    cell = Cell(
                        bbox=box.tolist(),
                        row=0,  # Temporary value
                        col=0,  # Temporary value
                        confidence=score.item()
                    )
                    cells.append(cell)
                    
            # Limit cell count
            if len(cells) > self.config['max_cells']:
                self.logger.warning(f"Cell count {len(cells)} exceeds maximum {self.config['max_cells']}, truncating")
                cells = sorted(cells, key=lambda x: x.confidence, reverse=True)[:self.config['max_cells']]
            
            # Process cells
            cells = self._assign_rows_cols(cells)
            cells = self._detect_merged_cells(cells)
            
            # Create table structure
            structure = TableStructure(
                cells=cells,
                rows=max(c.row + c.rowspan for c in cells),
                cols=max(c.col + c.colspan for c in cells),
                header_rows=self._detect_header_rows(cells)
            )
            
            # Log final structure
            self.logger.debug(f"Final table structure:", {
                'total_cells': len(structure.cells),
                'rows': structure.rows,
                'columns': structure.cols,
                'header_rows': structure.header_rows,
                'processing_time': f"{time.time() - start_time:.2f}s"
            })
            
            # Visualize structure if in debug mode
            if self.logger._logger.level == logging.DEBUG:
                self._visualize_structure(image, structure, file_path)
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Structure recognition failed: {str(e)}")
            return TableStructure(cells=[], rows=0, cols=0)  


            
    def _postprocess_cells(self, cells: List[Cell], image: Image.Image) -> List[Cell]:
        """Post-process and optimize cells
        
        Args:
            cells: List of cells
            image: Input image
            
        Returns:
            Optimized list of cells
        """
        try:
            processed_cells = []
            for cell in cells:
                x1, y1, x2, y2 = cell.bbox
                
                # Validate cell size
                width = x2 - x1
                height = y2 - y1
                
                if (width < self.config['min_cell_size'] or 
                    height < self.config['min_cell_size'] or
                    width > self.config['max_cell_size'] or
                    height > self.config['max_cell_size']):
                    continue
                    
                processed_cells.append(cell)
                
            return processed_cells
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Cell post-processing"
            })
            return cells
            
    def _assign_rows_cols(self, cells: List[Cell]) -> List[Cell]:
        """Assign rows and columns
        
        Args:
            cells: List of cells
            
        Returns:
            Updated list of cells with row and column information
        """
        try:
            # Extract cell centers
            centers = np.array([
                [(cell.bbox[0] + cell.bbox[2]) / 2, (cell.bbox[1] + cell.bbox[3]) / 2]
                for cell in cells
            ])
            
            # Use K-means clustering to determine rows and columns
            x_coords = centers[:, 0].reshape(-1, 1)
            y_coords = centers[:, 1].reshape(-1, 1)
            
            # Column clustering
            col_kmeans = KMeans(
                n_clusters=min(len(cells), 20),
                random_state=42
            )
            col_labels = col_kmeans.fit_predict(x_coords)
            
            # Row clustering
            row_kmeans = KMeans(
                n_clusters=min(len(cells), 20),
                random_state=42
            )
            row_labels = row_kmeans.fit_predict(y_coords)
            
            # Update cell row and column information
            for cell, row_label, col_label in zip(cells, row_labels, col_labels):
                cell.row = int(row_label)
                cell.col = int(col_label)
                
            return cells
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Row and column assignment"
            })
            raise
            
    def _detect_merged_cells(self, cells: List[Cell]) -> List[Cell]:
        """Detect merged cells
        
        Args:
            cells: List of cells
            
        Returns:
            Updated list of cells with merged cell information
        """
        try:
            # Sort by row and column
            cells.sort(key=lambda x: (x.row, x.col))
            
            # Create grid
            max_row = max(cell.row for cell in cells)
            max_col = max(cell.col for cell in cells)
            grid = [[None for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            
            # Fill grid
            for cell in cells:
                grid[cell.row][cell.col] = cell
                
            # Detect merged cells
            for row in range(max_row + 1):
                for col in range(max_col + 1):
                    if grid[row][col] is None:
                        continue
                        
                    cell = grid[row][col]
                    
                    # Check row merging
                    colspan = 1
                    while (col + colspan <= max_col and 
                           grid[row][col + colspan] is None):
                        colspan += 1
                        
                    # Check column merging
                    rowspan = 1
                    while (row + rowspan <= max_row and 
                           grid[row + rowspan][col] is None):
                        rowspan += 1
                        
                    # Update merging information
                    if colspan > 1 or rowspan > 1:
                        cell.rowspan = rowspan
                        cell.colspan = colspan
                        cell.cell_type = 'merged'
                        
            return cells
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Merged cell detection"
            })
            raise
            
    def _detect_header_rows(self, cells: List[Cell]) -> int:
        """Detect header rows
        
        Args:
            cells: List of cells
            
        Returns:
            Number of header rows
        """
        try:
            # Group by row
            rows = {}
            for cell in cells:
                if cell.row not in rows:
                    rows[cell.row] = []
                rows[cell.row].append(cell)
                
            # Calculate features for each row
            row_features = []
            for row in sorted(rows.keys()):
                cells_in_row = rows[row]
                
                # Calculate features
                header_score = 0
                for cell in cells_in_row:
                    # Check if at top
                    if cell.row == 0:
                        header_score += 1
                    # Check if spans rows
                    if cell.rowspan > 1:
                        header_score += 0.5
                    # Check if spans columns
                    if cell.colspan > 1:
                        header_score += 0.3
                        
                row_features.append((row, header_score))
                
            # Determine number of header rows
            header_rows = 0
            for row, score in row_features:
                if score > 1.0:  # Threshold can be adjusted
                    header_rows += 1
                else:
                    break
                    
            return header_rows
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "Header row detection"
            })
            return 0 
        

    def _draw_structure(self, image: Image.Image, structure: TableStructure) -> Image.Image:
        """Draw table structure
        
        Args:
            image: Input image
            structure: Table structure
            
        Returns:
            Marked image
        """
        # Create a copy of the image for drawing
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Draw cells
        for idx, cell in enumerate(structure.cells):
            # Draw cell boundary
            draw.rectangle(
                cell.bbox,
                outline='red',
                width=2
            )
            
            # Add cell info
            info = f"({cell.row},{cell.col})"
            if cell.rowspan > 1 or cell.colspan > 1:
                info += f" [{cell.rowspan}x{cell.colspan}]"
            draw.text(
                (cell.bbox[0] + 5, cell.bbox[1] + 5),
                info,
                fill='blue'
            )
            
            # Add confidence
            draw.text(
                (cell.bbox[0] + 5, cell.bbox[1] + 25),
                f"conf: {cell.confidence:.2f}",
                fill='green'
            )
            
        return draw_image

    def _visualize_structure(
        self,
        image: Image.Image,
        structure: TableStructure,
        file_path: Optional[str] = None
    ):
        """Visualize table structure
        Args:
            image: Input image
            structure: Table structure
            file_path: Optional file path for saving
        """
        try:
            # Create visualization
            draw_image = self._draw_structure(image, structure)
            # generate pdf_stem
            pdf_stem = None
            if file_path:
                pdf_stem = os.path.splitext(os.path.basename(file_path))[0]
            else:
                pdf_stem = "unknown"
            #generate filename with time 
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"structure_{timestamp}.png"
            #get output path 
            output_path = self.params.get('output_path', '')
            if not output_path:
                output_path = getattr(self.logger, 'config', {}).get('output_path', '')
            if not output_path:
                raise ValueError("Output path must be specified")
            # get_output_subpath
            debug_img_path = get_output_subpath(
                output_path,
                'debug',
                filename,
                pdf_stem
            )
            draw_image.save(debug_img_path)
            self.logger.info(f"Saved structure visualization: {debug_img_path}")
        except Exception as e:
            self.logger.error(f"Failed to visualize structure: {str(e)}")