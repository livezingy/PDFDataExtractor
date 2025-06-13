# core/processing/text_processor.py
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import time
from core.processing.base_processor import BaseProcessor
from core.processing.table_processor import TableProcessor
from core.processing.came_optimizer import CameOptimizer
from core.utils.logger import AppLogger
import os
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
import numpy as np
from core.utils.path_utils import get_output_subpath

class TextProcessor(BaseProcessor):
    """Text processor class
    
    Responsible for text extraction and table recognition in documents.
    Integrates OCR and table processing capabilities.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize text processor
        
        Args:
            params: Processing parameters
        """
        super().__init__()
        self.params = params or {}
        self.config = {
            'ocr': {
                'lang': 'eng',
                'preprocessing': True
            },
            'table': {
                'min_cell_area': 100,
                'max_cell_area': 10000,
                'min_cell_count': 4,
                'max_cell_count': 100
            },
            'overlap_threshold': 0.7,
            # Remove hardcoded camelot_accuracy_threshold, will use params
        }
        if params:
            self.config.update(params)
            
        self.table_processor = TableProcessor(self.config.get('table'))
        self.optimizer = CameOptimizer()
        # page image cache, (file_path, page_num)
        self._page_image_cache = {}


    def process(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process PDF or image file
        
        Args:
            file_path: Path to PDF or image file
            params: Processing parameters
            
        Returns:
            Processing results
        """
        try:
            params = params.copy() if params else self.params.copy()
            file_path = params.get('current_filepath')
            self.logger.info(f"Starting document processing: {file_path}")
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                # Image file: use transformer+OCR pipeline
                return self._process_image_file(file_path, params)
            
            # PDF file processing
            pages = params.get('pages', 'all')
            page_list = []
            if pages == 'all':
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        page_list = list(range(1, len(pdf.pages) + 1))
                except Exception as e:
                    self.logger.error(f"Failed to get total pages: {str(e)}")
                    return {'success': False, 'error': str(e), 'pages': {}, 'tables': []}
            else:
                # support 1-3,5
                try:
                    page_list = []
                    for part in str(pages).split(','):
                        part = part.strip()
                        if '-' in part:
                            start, end = part.split('-')
                            page_list.extend(list(range(int(start), int(end)+1)))
                        elif part:
                            page_list.append(int(part))
                    page_list = sorted(set(page_list))
                except Exception as e:
                    self.logger.error(f"Invalid pages parameter: {pages}, error: {str(e)}")
                    return {'success': False, 'error': f'Invalid pages parameter: {pages}', 'pages': {}, 'tables': []}
                
            results = {'tables': [], 'pages': {}}
            for page_num in page_list:
                try:
                    page_result = self.process_page(page_num, params)
                    if page_result['success']:
                        results['pages'][str(page_num)] = page_result['tables']
                        results['tables'].extend(page_result['tables'])
                    self._page_image_cache.clear() # Clear cache after each page to save memory
                except Exception as e:
                    self.logger.error(f"Page {page_num} processing failed: {str(e)}")
                    continue
            # Export results if needed
            if params.get('export_results', True) and results['tables']:
                try:
                    export_path = self._handle_export(results, params)
                    results['export_path'] = export_path
                except Exception as e:
                    self.logger.error(f"Export failed: {str(e)}")
            return {
                'success': True,
                'pages': results['pages'],
                'tables': results['tables'],
                'export_path': results.get('export_path', '')
            }
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'pages': {},
                'tables': []
            }

    def _process_image_file(self, file_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single image file using transformer detection + structure + OCR"""
        try:
            image = Image.open(file_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            detected_regions = self.table_processor.detect_tables(image, {'pages': 1})
            self.logger.debug(f"Detected {len(detected_regions)} regions in image {file_path}")
            # Visualize detection results
            self._visualize_detection_results(
                image,
                [],  # there is no camelot results for image
                detected_regions,
                1
            )
            tables = []
            for region in detected_regions:
                table = self.table_processor.process_region(
                    image,
                    region,
                    {**params, 'page_num': 1}
                )
                if table:
                    tables.append(table)
            
            results = {
                'success': True,
                'pages': {'1': tables},
                'tables': tables,
                'export_path': ''
            }
            if params.get('export_results', True) and tables:
                try:
                    export_path = self._handle_export(results, params)
                    results['export_path'] = export_path
                except Exception as e:
                    self.logger.error(f"Export failed: {str(e)}")
            return results
        except Exception as e:
            self.logger.error(f"Image file processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'pages': {},
                'tables': []
            }

    def process_page(self, page_num: int, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            params_copy = params.copy()
            camelot_results, detected_regions = self._get_parallel_results(page_num, params_copy)
            page_tables = self._process_detected_regions(page_num, params_copy, camelot_results, detected_regions)
            return {
                'success': True,
                'tables': page_tables
            }
        except Exception as e:
            self.logger.error(f"Page {page_num} processing failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'tables': []
            }

    def _get_parallel_results(self, page_num: int, params: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        try:
            params_copy = params.copy()
            file_path = params_copy.get('current_filepath')
            pdf_size = None
            with pdfplumber.open(file_path) as pdf:
                page = pdf.pages[page_num - 1]
                pdf_size = (page.width, page.height)
                self.logger.debug(f"PDF page size for {file_path} page {page_num}: {pdf_size}")

            process_image = self._load_page_image(file_path, page_num)
            self.logger.debug(f"Loaded page image for {file_path} page {page_num}, size: {process_image.size}")

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.logger.debug(f"Starting Camelot processing for {file_path} page {page_num}")
                camelot_future = executor.submit(
                    self.optimizer.optimize_table,
                    params_copy
                )
                self.logger.debug(f"Starting Transformer detection for {file_path} page {page_num}")
                model_future = executor.submit(
                    self.table_processor.detect_tables,
                    process_image,
                    params_copy
                )
                camelot_tables = camelot_future.result()
                detected_regions = model_future.result()

                # Robust None/type checks
                if camelot_tables is None or not isinstance(camelot_tables, (list, tuple)):
                    camelot_tables = []
                if detected_regions is None or not isinstance(detected_regions, (list, tuple)):
                    detected_regions = []

                if camelot_tables and process_image.size:
                    image_to_pdf_scale = (
                        process_image.size[0] / pdf_size[0],
                        process_image.size[1] / pdf_size[1]
                    )
                    self.logger.debug(f"Image to PDF scale factor: {image_to_pdf_scale}")

                self.logger.debug(f"Camelot detection results for {file_path} page {page_num}:", {
                    'table_count': len(camelot_tables),
                    'tables': [
                        {
                            'accuracy': getattr(table, 'accuracy', None),
                            'whitespace': getattr(table, 'whitespace', None),
                            'bbox': getattr(table, '_bbox', None)
                        }
                        for table in camelot_tables
                    ]
                })
                self.logger.debug(f"Transformer detection results for {file_path} page {page_num}:", {
                    'table_count': len(detected_regions),
                    'regions': [
                        {
                            'confidence': region.get('confidence'),
                            'bbox': region.get('bbox'),
                            'type': region.get('type')
                        }
                        for region in detected_regions
                    ]
                })
                camelot_results = []
                if camelot_tables:
                    camelot_results = [
                        {
                            'data': getattr(table, 'df', None).values.tolist() if getattr(table, 'df', None) is not None else [],
                            'columns': getattr(table, 'df', None).columns.tolist() if getattr(table, 'df', None) is not None else [],
                            'accuracy': getattr(table, 'accuracy', None),
                            'whitespace': getattr(table, 'whitespace', None),
                            'bbox': getattr(table, '_bbox', None),
                            'page': page_num,
                            'pdf_size': pdf_size,
                            'matched': False,
                            'accuracy_acceptable': False
                        }
                        for table in camelot_tables
                    ]
                
                self._visualize_detection_results(
                    process_image,
                    camelot_results,
                    detected_regions,
                    page_num
                )
                return camelot_results, detected_regions
        except Exception as e:
            self.logger.error(f"Parallel processing failed for page {page_num}: {str(e)}", exc_info=True)
            return [], []

    def _process_detected_regions(self, page_num: int, params: Dict[str, Any], camelot_results: List[Dict], detected_regions: List[Dict]) -> List[Dict]:
        """
        1. For each table-transformer detected region, find the matching Camelot region (after coordinate transformation), and decide to use Camelot, retry Camelot with tuned parameters, or use table-transformer+OCR based on accuracy thresholds.
        2. For regions not matched to Camelot, use table-transformer+OCR.
        3. Deduplicate results to ensure each table is output only once.
        """
        page_tables = []        
        threshold_camlot = params.get('camelot_accuracy_threshold', 0.6)  # Use user param
        used_camelot_idx = set()
        file_path = params.get('current_filepath', '')
        process_image = self._load_page_image(file_path, page_num)

        # 1. Iterate over table-transformer detected regions
        for region in detected_regions:
            region_bbox = region['bbox']
            best_camelot = None
            best_overlap = 0
            best_idx = -1
            image_width, image_height = process_image.size
            for idx, camelot_table in enumerate(camelot_results):
                if idx in used_camelot_idx:
                    continue
                camelot_bbox = camelot_table['bbox']
                if 'pdf_size' in camelot_table:
                    pdf_width, pdf_height = camelot_table['pdf_size']
                    x1, y2 = self.pdf_to_image_coords(camelot_bbox[0], camelot_bbox[1], image_width, image_height, pdf_width, pdf_height)
                    x2, y1 = self.pdf_to_image_coords(camelot_bbox[2], camelot_bbox[3], image_width, image_height, pdf_width, pdf_height)
                    camelot_bbox_img = [x1, y1, x2, y2]
                else:
                    camelot_bbox_img = camelot_bbox
                overlap = self._calculate_overlap(region_bbox, camelot_bbox_img)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_camelot = camelot_table
                    best_idx = idx
            # 3. Decide based on accuracy
            if best_overlap > self.config.get('overlap_threshold', 0.7):
                used_camelot_idx.add(best_idx)
                acc = best_camelot['accuracy']
                if acc >= threshold_camlot:
                    # 3.1 Use Camelot result
                    best_camelot['matched'] = True
                    best_camelot['accuracy_acceptable'] = True
                    page_tables.append(best_camelot)
                    self.logger.debug(f"[Camelot] Used high-accuracy result for region {region_bbox} acc={acc}")
               
                else:
                    # 3.2 Camelot accuracy too low, fallback to Transformer+OCR
                    self.logger.debug(f"[Camelot] Low accuracy, fallback to Transformer+OCR for region {region_bbox} acc={acc}")
                    table = self._process_with_model(page_num, region, params)
                    if table:
                        page_tables.append(table)
            else:
                # 4. No matching Camelot region, use Transformer+OCR
                self.logger.debug(f"[Transformer] No matching Camelot, use Transformer+OCR for region {region_bbox}")
                table = self._process_with_model(page_num, region, params)
                if table:
                    page_tables.append(table)
        # 5. Add unmatched high-accuracy Camelot tables
        for idx, camelot_table in enumerate(camelot_results):
            if idx not in used_camelot_idx and camelot_table['accuracy'] >= threshold_camlot:
                camelot_table['matched'] = True
                camelot_table['accuracy_acceptable'] = True
                page_tables.append(camelot_table)
        return page_tables

    def _process_with_model(self, page_num: int, region: Dict, params: Dict[str, Any]) -> Optional[Dict]:
        try:
            file_path = params.get('current_filepath', '')
            image = params.get('process_image')
            if image is None:
                image = self._load_page_image(file_path, page_num)
            table = self.table_processor.process_region(
                image,
                region,
                {**params, 'page_num': page_num}
            )
            return table
        except Exception as e:
            self.logger.error(f"Model processing failed for {file_path} page {page_num}: {str(e)}")
            return None

    def _load_page_image(self, file_path: str, page_num: int) -> Image.Image:
    
        cache_key = (file_path, page_num)
        if cache_key in self._page_image_cache:
            self.logger.debug(f"Using cached image for {file_path} page {page_num}")
            return self._page_image_cache[cache_key]
        try:
            import pdf2image
            images = pdf2image.convert_from_path(
                file_path,
                first_page=page_num,
                last_page=page_num
            )
            image = images[0]
            self._page_image_cache[cache_key] = image
            self.logger.debug(f"Loaded and cached image for {file_path} page {page_num}, size: {image.size}")
            return image
        except Exception as e:
            self.logger.error(f"Failed to load page image: {str(e)}")
            raise RuntimeError(f"Failed to load page image: {str(e)}")
        

    def _calculate_overlap(self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate overlap ratio between two bounding boxes
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        self.logger.debug(f"Calculating overlap between boxes:", {
            'box1': {
                'x1': x1_1,
                'y1': y1_1,
                'x2': x2_1,
                'y2': y2_1,
                'width': x2_1 - x1_1,
                'height': y2_1 - y1_1
            },
            'box2': {
                'x1': x1_2,
                'y1': y1_2,
                'x2': x2_2,
                'y2': y2_2,
                'width': x2_2 - x1_2,
                'height': y2_2 - y1_2
            }
        })
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            self.logger.debug("No overlap detected")
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate overlap ratio (intersection over union)
        union_area = area1 + area2 - intersection_area
        overlap_ratio = intersection_area / union_area if union_area > 0 else 0.0
        
        self.logger.debug(f"Overlap calculation results:", {
            'intersection_area': intersection_area,
            'area1': area1,
            'area2': area2,
            'union_area': union_area,
            'overlap_ratio': overlap_ratio
        })
        
        return overlap_ratio
    

    def _find_best_matching_table(
        self,
        bbox: Tuple[float, float, float, float],
        camelot_tables: List[Dict],
        image_width: int,
        image_height: int,
        params: Dict[str, Any]
    ) -> Optional[Dict]:
        """Find best matching Camelot table
        
        Args:
            bbox: Table bounding box (in image coordinates)
            camelot_tables: List of Camelot tables
            image_width: Width of the page image
            image_height: Height of the page image
            params: Processing parameters
            
        Returns:
            Best matching table or None
        """
        best_table = None
        best_overlap = 0
        
        self.logger.debug(f"Finding best matching table for bbox: {bbox}")
        
        # Get PDF page height for coordinate conversion
        pdf_height = None
        pdf_width = None
        if camelot_tables and 'pdf_size' in camelot_tables[0]:
            pdf_width, pdf_height = camelot_tables[0]['pdf_size']
            self.logger.debug(f"PDF size for coordinate conversion: {pdf_width}x{pdf_height}")
        for idx, camelot_table in enumerate(camelot_tables):
            camelot_bbox = camelot_table['bbox']
            original_bbox = list(camelot_bbox)
            if pdf_height and pdf_width and image_width and image_height:
                x1, y2 = self.pdf_to_image_coords(camelot_bbox[0], camelot_bbox[1], image_width, image_height, pdf_width, pdf_height)
                x2, y1 = self.pdf_to_image_coords(camelot_bbox[2], camelot_bbox[3], image_width, image_height, pdf_width, pdf_height)
                camelot_bbox = [x1, y1, x2, y2]
                self.logger.debug(f"Coordinate conversion for table {idx}:", {
                    'original_bbox': original_bbox,
                    'converted_bbox': camelot_bbox,
                    'pdf_size': (pdf_width, pdf_height),
                    'image_size': (image_width, image_height)
                })
            overlap = self._calculate_overlap(
                bbox,
                camelot_bbox
            )
            
            self.logger.debug(f"Overlap calculation for table {idx}:", {
                'table_accuracy': camelot_table.get('accuracy', 0),
                'overlap_ratio': overlap,
                'current_best': best_overlap
            })
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_table = camelot_table
                self.logger.debug(f"New best match found:", {
                    'table_idx': idx,
                    'overlap_ratio': overlap,
                    'table_accuracy': camelot_table.get('accuracy', 0)
                })
                
        threshold = self.config.get('overlap_threshold', 0.7)
        self.logger.debug(f"Final matching result:", {
            'best_overlap': best_overlap,
            'threshold': threshold,
            'matched': best_overlap > threshold
        })
                
        if best_overlap > threshold:
            return best_table
            
        return None
            
   
    
    def _visualize_detection_results(
        self,
        image: Image.Image,
        camelot_results: List[Dict],
        transformer_results: List[Dict],
        page_num: int
    ):
        """Visualize detection results and save annotated image
        
        Args:
            image: Input image
            camelot_results: Camelot detection results
            transformer_results: Transformer detection results
            file_path: Original file path
            page_num: Page number
        """
        try:
            from PIL import ImageDraw, ImageFont
            # check if image is valid
            if image is None or not hasattr(image, 'copy'):
                self.logger.error("Input image is None or invalid, cannot visualize detection results.")
                return
            # check if results are lists            
            if not isinstance(camelot_results, list):
                camelot_results = []
            if not isinstance(transformer_results, list):
                transformer_results = []
            if not camelot_results and not transformer_results:
                self.logger.info("No detection results to visualize.")
                return

            
            self.logger.debug(f"Visualizing detection results for page {page_num} with {len(camelot_results)} Camelot tables and {len(transformer_results)} Transformer regions.")
            # Create a copy of the image for drawing
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            # Load font
            font = None
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
            # Draw Camelot results in blue
            for idx, result in enumerate(camelot_results):
                bbox = result.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue
                # Convert coordinates if needed
                if 'pdf_size' in result:
                    pdf_width, pdf_height = result['pdf_size']
                    image_width, image_height = image.size
                    x1, y2 = self.pdf_to_image_coords(bbox[0], bbox[1], image_width, image_height, pdf_width, pdf_height)
                    x2, y1 = self.pdf_to_image_coords(bbox[2], bbox[3], image_width, image_height, pdf_width, pdf_height)
                    bbox = [x1, y1, x2, y2]
                    self.logger.debug(f"Camelot table {idx+1} converted bbox: {bbox}")
                draw.rectangle(bbox, outline='blue', width=3)
                try:
                    acc = float(result.get('accuracy', 0))
                except Exception:
                    acc = 0
                draw.text((bbox[0], bbox[1] - 28), f"Camelot {idx+1} (acc: {acc:.2f})", fill='blue', font=font)
            # Draw Transformer results in red
            for idx, result in enumerate(transformer_results):
                bbox = result.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue
                draw.rectangle(bbox, outline='red', width=3)
                try:
                    conf = float(result.get('confidence', 0))
                except Exception:
                    conf = 0
                draw.text((bbox[0], bbox[1] - 28), f"Transformer {idx+1} (conf: {conf:.2f})", fill='red', font=font)
            filename = f"page{page_num}_detection.png"
            output_path = get_output_subpath(self.params, 'preview', filename=filename)
            # Save annotated image
            draw_image.save(output_path)
            self.logger.info(f"Saved detection visualization to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to visualize detection results: {str(e)}", exc_info=True)


    def get_coordinate_transformers(self, pdf_image, pdf_width, pdf_height):
        image_width, image_height = pdf_image.size
        
        image_scalers = (
            image_width / float(pdf_width),
            image_height / float(pdf_height)
        )
        
        pdf_scalers = (
            pdf_width / float(image_width),
            pdf_height / float(image_height)
        )        
        return image_scalers, pdf_scalers
    
        
    def pdf_to_image_coords(self, x, y, image_width, image_height, pdf_width, pdf_height):
        """Convert PDF coordinates to image coordinates using explicit image/pdf size."""
        scale_x = image_width / float(pdf_width)
        scale_y = image_height / float(pdf_height)
        img_x = int(x * scale_x)
        img_y = int(abs(y - pdf_height) * scale_y)
        return img_x, img_y

