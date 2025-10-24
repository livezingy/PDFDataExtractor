# core/processing/page_processor.py
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import time, os, io
from core.processing.base_processor import BaseProcessor
from core.processing.table_processor import TableProcessor
from core.utils.logger import AppLogger
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
import numpy as np
from core.utils.path_utils import get_output_subpath
import fitz  # PyMuPDF for image extraction

class PageProcessor(BaseProcessor):

    """Page processor class

    Responsible for text extraction and table recognition in documents.
    Integrates OCR and table processing capabilities.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize text processor
        
        Args:
            params: Processing parameters
        """
        super().__init__()
        self.logger = AppLogger.get_logger()
        # 不再需要 self.config
        self.params = params or {}

    def _is_scanned_pdf(self, pdfplumber_object, page_num: int) -> bool:
        """检测PDF页面是否为扫描文档
        
        Args:
            pdfplumber_object: pdfplumber PDF对象
            page_num: 页面编号（从1开始）
            
        Returns:
            bool: True表示扫描PDF，False表示文本PDF
        """
        try:
            page = pdfplumber_object.pages[page_num - 1]
            text = page.extract_text()
            
            # 如果文本长度少于50个字符，认为是扫描PDF
            if text is None or len(text.strip()) < 50:
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting scanned PDF for page {page_num}: {e}")
            # 出错时默认认为是文本PDF
            return False

    async def _process_with_transformer(self, image, params):
        """使用Transformer处理图像
        
        Args:
            image: PIL Image对象
            params: 处理参数
            
        Returns:
            dict: 处理结果
        """
        table_parser = params.get('table_parser')
        
        # 如果没有提供table_parser，尝试创建
        if not table_parser:
            try:
                from core.models.table_parser import TableParser
                from core.utils.config import load_config
                app_config = load_config()
                table_parser = TableParser(app_config)
                self.logger.info("[PageProcessor] Created TableParser instance")
            except Exception as e:
                self.logger.error(f"[PageProcessor] Failed to create TableParser: {str(e)}")
                return {'success': False, 'error': f'Failed to create TableParser: {str(e)}', 'tables': []}
        
        if table_parser and hasattr(table_parser, 'parser_image'):
            self.logger.info("[PageProcessor] Using Transformer for image processing")
            try:
                if not hasattr(table_parser, 'models') or table_parser.models is None:
                    self.logger.error("[PageProcessor] TableParser models not initialized")
                    return {'success': False, 'error': 'TableParser models not initialized', 'tables': []}
                else:
                    results = await table_parser.parser_image(image, params)
                    return results
            except Exception as e:
                self.logger.error(f"[PageProcessor] Error calling table_parser.parser_image: {str(e)}")
                return {'success': False, 'error': str(e), 'tables': []}
        else:
            self.logger.error("[PageProcessor] table_parser not found in params or missing parser_image method")
            return {'success': False, 'error': 'table_parser not available', 'tables': []}

    async def _process_page_as_scanned(self, page, params):
        """将PDF页面转为图像后用Transformer处理
        
        Args:
            page: pdfplumber页面对象
            params: 处理参数
            
        Returns:
            dict: 处理结果
        """
        try:
            # 将PDF页面转换为图像，提高分辨率以获得更好的OCR效果
            image = page.to_image(resolution=300)  # 提高到300 DPI
            pil_image = Image.frombytes('RGB', image.original.size, image.original.tobytes())
            
            self.logger.info(f"[PageProcessor] Converted PDF page to image for Transformer processing (resolution: 300 DPI, size: {pil_image.size})")
            return await self._process_with_transformer(pil_image, params)
            
        except Exception as e:
            self.logger.error(f"[PageProcessor] Error converting page to image: {str(e)}")
            return {'success': False, 'error': str(e), 'tables': []}

    async def _process_with_auto_flavor(self, page, method, params):
        """Auto模式：自动选择flavor
        
        Args:
            page: pdfplumber页面对象
            method: 提取方法
            params: 处理参数
            
        Returns:
            dict: 处理结果
        """
        from core.processing.table_processor import PageFeatureAnalyzer
        
        try:
            # 创建特征分析器
            analyzer = PageFeatureAnalyzer(page)
            
            # 预测表格类型
            table_type = analyzer.predict_table_type()
            
            # 根据方法和类型选择flavor
            if method == 'camelot':
                flavor = 'lattice' if table_type == 'bordered' else 'stream'
            elif method == 'pdfplumber':
                flavor = 'lines' if table_type == 'bordered' else 'text'
            else:
                flavor = 'auto'  # 保持原值
            
            self.logger.info(f"[PageProcessor] Auto模式：检测到{table_type}表格，选择{flavor}方法")
            
            # 调用标准处理
            return await self._process_with_method(page, method, flavor, params)
            
        except Exception as e:
            self.logger.error(f"[PageProcessor] Error in auto flavor selection: {str(e)}")
            return {'success': False, 'error': str(e), 'tables': []}

    async def _process_with_method(self, page, method, flavor, params):
        """使用指定方法和flavor处理页面
        
        Args:
            page: pdfplumber页面对象
            method: 提取方法
            flavor: 提取flavor
            params: 处理参数
            
        Returns:
            dict: 处理结果
        """
        try:
            table_processor = TableProcessor(params)
            pdf_path = params.get('current_filepath')
            
            if method == 'camelot':
                page_num = getattr(page, 'page_number', 1)
                if flavor == 'lattice':
                    results = table_processor.extract_camelot_lattice(pdf_path, page_num, page)
                elif flavor == 'stream':
                    results = table_processor.extract_camelot_stream(pdf_path, page_num, page)
                else:
                    self.logger.error(f"[PageProcessor] Unknown Camelot flavor: {flavor}")
                    return {'success': False, 'error': f'Unknown Camelot flavor: {flavor}', 'tables': []}
            elif method == 'pdfplumber':
                if flavor == 'lines':
                    results = table_processor.extract_pdfplumber_lines(page)
                elif flavor == 'text':
                    results = table_processor.extract_pdfplumber_text(page)
                else:
                    self.logger.error(f"[PageProcessor] Unknown PDFPlumber flavor: {flavor}")
                    return {'success': False, 'error': f'Unknown PDFPlumber flavor: {flavor}', 'tables': []}
            elif method == 'transformer':
                # Transformer需要将页面转为图像
                return await self._process_page_as_scanned(page, params)
            else:
                self.logger.error(f"[PageProcessor] Unknown method: {method}")
                return {'success': False, 'error': f'Unknown method: {method}', 'tables': []}
            
            return {'success': True, 'tables': results, 'error': ''}
            
        except Exception as e:
            self.logger.error(f"[PageProcessor] Error in method processing: {str(e)}")
            return {'success': False, 'error': str(e), 'tables': []}

    async def process(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process PDF or image file"""
        try:
            params = params.copy() if params else self.params.copy()
            file_path = params.get('current_filepath')
            self.logger.info(f"[PageProcessor] Starting document processing: {file_path}")
            ext = os.path.splitext(file_path)[1].lower()
            self.logger.debug(f"[PageProcessor] File extension: {ext}")
            self.logger.debug(f"[PageProcessor] Params: {params}")

            results = {'tables': [], 'error': "", 'pages': []}

            # 1. 处理图像文件
            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                self.logger.info("[PageProcessor] Detected image file, using Transformer")
                image = Image.open(file_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # 添加页面号信息（图像文件默认为第1页）
                params['current_page_num'] = 1
                results = await self._process_with_transformer(image, params)
                
                # Export results if needed (same logic as PDF processing)
                if params.get('export_results', True) and results.get('tables'):
                    try:
                        self.logger.info("[PageProcessor] Exporting Transformer results")
                        export_path = self._handle_export(results, params)
                        results['export_path'] = export_path
                    except Exception as e:
                        self.logger.error(f"[PageProcessor] Transformer export failed: {str(e)}")
                
                return {
                    'success': results.get('success', True),
                    'tables': results.get('tables', []) if isinstance(results, dict) else [],
                    'export_path': results.get('export_path', None),
                    'pages': [],
                    'error': results.get('error', '') if isinstance(results, dict) else ''
                }

            # 2. 处理PDF文件
            self.logger.info("[PageProcessor] Detected PDF file, starting pdfplumber open")
            pages = params.get('pages', 'all')
            # Treat empty or whitespace pages as 'all'
            if pages is None or str(pages).strip() == "":
                pages = 'all'
            
            page_list = []
            try:
                with pdfplumber.open(file_path) as pdfplumber_object:
                    self.logger.info("[PageProcessor] PDF opened successfully")
                    if not hasattr(pdfplumber_object, 'pages') or not pdfplumber_object.pages:
                        self.logger.error("No pages found in PDF.")
                        return {
                            'success': False,
                            'tables': [],
                            'export_path': None,
                            'pages': [],
                            'error': "No pages found in PDF."
                        }
                    
                    # 解析页面列表
                    if pages == 'all':
                        try:
                            page_list = list(range(1, len(pdfplumber_object.pages) + 1))
                            self.logger.info(f"[PageProcessor] Processing all pages: {page_list}")
                        except Exception as e:
                            self.logger.error(f"[PageProcessor] Failed to get total pages: {str(e)}")
                            return {
                                'success': False,
                                'tables': [],
                                'export_path': None,
                                'pages': [],
                                'error': str(e)
                            }
                    else:
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
                            if not page_list:
                                # Fallback to all pages when custom input is empty/invalid
                                page_list = list(range(1, len(pdfplumber_object.pages) + 1))
                                self.logger.info("[PageProcessor] Empty custom pages, fallback to all pages")
                            else:
                                self.logger.info(f"[PageProcessor] Custom page list: {page_list}")
                        except Exception as e:
                            self.logger.error(f"[PageProcessor] Invalid pages parameter: {pages}, error: {str(e)}")
                            return {
                                'success': False,
                                'tables': [],
                                'export_path': None,
                                'pages': [],
                                'error': f'Invalid pages parameter: {pages}'
                            }

                    # 处理每个页面
                    for page_num in page_list:
                        try:
                            self.logger.info(f"[PageProcessor] Processing page {page_num}")
                            page = pdfplumber_object.pages[page_num - 1]
                            
                            # 添加当前页面号到params
                            page_params = params.copy()
                            page_params['current_page_num'] = page_num
                            
                            # 3. 检测是否为扫描PDF
                            if self._is_scanned_pdf(pdfplumber_object, page_num):
                                # 记录警告日志
                                self.logger.warning(f"检测到扫描文档（页面{page_num}），自动使用Transformer提取，用户设定的方法无效")
                                # 转换为图像并使用Transformer
                                page_result = await self._process_page_as_scanned(page, page_params)
                            else:
                                # 4. 使用用户选择的方法
                                method = params.get('table_method', 'camelot')
                                flavor = params.get('table_flavor', 'auto')
                                
                                if flavor == 'auto':
                                    # 调用predict_table_type判断
                                    page_result = await self._process_with_auto_flavor(page, method, page_params)
                                else:
                                    # 直接使用指定方法和flavor
                                    page_result = await self._process_with_method(page, method, flavor, page_params)
                            
                            if page_result.get('success', True):
                                results['tables'].extend(page_result.get('tables', []))
                            else:
                                self.logger.error(f"[PageProcessor] Page {page_num} failed: {page_result.get('error', '')}")
                        except Exception as e:
                            self.logger.error(f"[PageProcessor] Page {page_num} processing failed: {str(e)}")
                            continue
            except Exception as e:
                self.logger.error(f"[PageProcessor] Failed to open PDF: {str(e)}")
                return {
                    'success': False,
                    'tables': [],
                    'export_path': None,
                    'pages': [],
                    'error': str(e)
                }

            # Save images if requested
            if params.get('save_images', False):
                try:
                    self.logger.info("[PageProcessor] Saving PDF images")
                    self._save_pdf_images(file_path, params)
                except Exception as e:
                    self.logger.error(f"[PageProcessor] Image saving failed: {str(e)}")

            # Export results if needed
            if params.get('export_results', True) and results['tables']:
                try:
                    self.logger.info("[PageProcessor] Exporting results")
                    export_path = self._handle_export(results, params)
                    results['export_path'] = export_path
                except Exception as e:
                    self.logger.error(f"[PageProcessor] Export failed: {str(e)}")
            self.logger.info("[PageProcessor] Processing finished")
            return {
                'success': True,
                'tables': results.get('tables', []),
                'export_path': results.get('export_path', None),
                'pages': page_list,
                'error': results.get('error', '')
            }
        except Exception as e:
            self.logger.error(f"[PageProcessor] Document processing failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'tables': [],
                'export_path': None,
                'pages': [],
                'error': str(e)
            }


    async def process_page(self, pdfplumber_object, page_num, params, table_processor):
        """
        Process a single page of the PDF document.
        Extracts tables from image and text regions separately in current page.
        table-transformer is used for image regions, traditional methods for text regions.
        
        Args:
            pdfplumber_object: the object returned by pdfplumber.open()
            page_num: the current page number (1-based)
            params: processing parameters
        Returns:
            dict: {'image_tables': [...], 'text_tables': [...], 'success': True/False, 'error': ...}
        """
        import threading
        results = {'image_tables': [], 'text_tables': [], 'success': True, 'error': ''}
        try:            
            page = pdfplumber_object.pages[page_num - 1]
            page_without_images = page
            # 1. extract all the images area (guard None)
            images_attr = getattr(page, 'images', None)
            images_list = images_attr if isinstance(images_attr, list) else []
            if images_list:
                image_bboxes = [(img['x0'], img['top'], img['x1'], img['bottom']) for img in images_list]
                images = []
                self.logger.debug(f"[process_page] found {len(image_bboxes)} image regions on page {page_num}")
                for bbox in image_bboxes:
                    cropped = page.within_bbox(bbox).to_image(resolution=300)
                    pil_img = cropped.original
                    images.append({'bbox': bbox, 'image': pil_img})
                #此处的bbox是否有意义？？？
                # 2. construct the page without images
                
                for bbox in image_bboxes:
                    page_without_images = page_without_images.outside_bbox(bbox)
                # 3. thread 1: process all image regions
                async def process_images():
                    img_params = params.copy()
                    for img_info in images:
                        img_params['image_bbox'] = img_info['bbox']
                        try:
                            table_parser = params.get('table_parser')
                            if table_parser and hasattr(table_parser, 'parser_image'):
                                try:
                                    # 检查table_parser的models是否正确初始化
                                    if not hasattr(table_parser, 'models') or table_parser.models is None:
                                        self.logger.error("[PageProcessor] TableParser models not initialized for image region")
                                        continue
                                    img_res = await table_parser.parser_image(img_info['image'], params)
                                    if img_res and isinstance(img_res, dict) and img_res.get('tables'):
                                        results['image_tables'].extend(img_res['tables'])
                                except Exception as e:
                                    self.logger.error(f"[PageProcessor] Error processing image region: {str(e)}")
                                    continue
                            else:
                                self.logger.warning("[PageProcessor] table_parser not available for image region processing")
                        except Exception as e:
                            self.logger.error(f"Image region table extraction failed: {str(e)}")

                await process_images()
            else:
                images = []
                self.logger.info(f"[process_page] No images found on page {page_num}, skipping image region processing.")


            # 4. thread 1: process all text regions
            def process_text():
                try:
                    # 使用传入的table_processor实例，避免重复创建
                    if not table_processor:
                        self.logger.error("TableProcessor instance is None.")
                        return
                    text_tables = table_processor.process_pdf_page(params.get('current_filepath'), page)
                    if not text_tables:
                        text_tables = []
                    results['text_tables'] = text_tables
                except Exception as e:
                    self.logger.error(f"Text region table extraction failed: {str(e)}")

            # Start the text processing thread
            t1 = threading.Thread(target=process_text)
            t1.start()
            t1.join()
            results['tables'] = []
            results['tables'].extend(results.get('image_tables', []))
            results['tables'].extend(results.get('text_tables', []))
            
            # Convert table format for export compatibility
            tables = results.get('tables')
            if tables is None:
                tables = []
            results['tables'] = self._convert_tables_for_export(tables)
            return results
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            self.logger.error(f"process_pdfplumber_page_with_images failed: {str(e)}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return results

    
    
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
                font = ImageFont.truetype("arial.ttf", 26)
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
                # Camelot: left-top对齐
                text = f"Camelot {idx+1} (acc: {acc:.2f})"
                draw.text((bbox[0], bbox[1] - 28), text, fill='blue', font=font)
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
                # Transformer: 左下角对齐
                text = f"Transformer {idx+1} (conf: {conf:.2f})"
                draw.text((bbox[0], bbox[3]), text, fill='red', font=font)
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

    def _save_pdf_images(self, pdf_path: str, params: Dict[str, Any]) -> None:
        """Save images from PDF to output/images/filename/ folder
        
        Args:
            pdf_path: Path to the PDF file
            params: Processing parameters containing output_path
        """
        try:
            # Get output path and create images directory
            output_path = params.get('output_path', '')
            if not output_path:
                self.logger.error("Output path not specified for image saving")
                return
                
            # Create images directory structure: output/images/filename/
            pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            images_dir = os.path.join(output_path, 'images', pdf_filename)
            os.makedirs(images_dir, exist_ok=True)
            
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)
            image_index = 0
            
            # Process each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Get images from the page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        # Skip if image is too small or invalid
                        if pix.width < 10 or pix.height < 10:
                            pix = None
                            continue
                            
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        # Create PIL Image
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Try to get image name from PDF metadata
                        image_name = None
                        try:
                            # Check if image has a name in the PDF
                            if hasattr(img, 'name') and img.name:
                                image_name = img.name
                            # Alternative: check if there's a title or alt text
                            elif hasattr(img, 'title') and img.title:
                                image_name = img.title
                        except:
                            pass
                        
                        # Generate filename
                        if image_name:
                            # Clean the image name for filesystem
                            safe_name = "".join(c for c in image_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                            safe_name = safe_name.replace(' ', '_')
                            if not safe_name:
                                safe_name = f"image_{image_index}"
                            filename = f"{safe_name}.png"
                        else:
                            # Use default naming: filename_page_index_image_index
                            filename = f"{pdf_filename}_page_{page_num + 1}_image_{img_index + 1}.png"
                        
                        # Save image
                        image_path = os.path.join(images_dir, filename)
                        pil_image.save(image_path, "PNG")
                        
                        self.logger.info(f"Saved image: {image_path}")
                        image_index += 1
                        
                        # Clean up
                        pix = None
                        
                    except Exception as e:
                        self.logger.error(f"Failed to save image {img_index} from page {page_num + 1}: {str(e)}")
                        continue
                        
            pdf_document.close()
            self.logger.info(f"Image saving completed. Images saved to: {images_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save PDF images: {str(e)}", exc_info=True)

    def _convert_tables_for_export(self, tables: List[Dict]) -> List[Dict]:
        """Convert table data to export-compatible format
        
        Args:
            tables: List of tables in various formats
            
        Returns:
            List of tables in export-compatible format with 'data' and 'columns' keys
        """
        converted_tables = []
        
        for table in tables:
            try:
                # Skip if already in correct format
                if isinstance(table, dict) and 'data' in table and 'columns' in table:
                    converted_tables.append(table)
                    continue
                
                # Convert table processor format
                if isinstance(table, dict) and 'table' in table:
                    table_obj = table['table']
                    
                    # Handle different table object types
                    if hasattr(table_obj, 'df'):  # Camelot table
                        df = table_obj.df
                        converted_table = {
                            'data': df.to_dict('records'),
                            'columns': df.columns.tolist(),
                            'confidence': table.get('score', 0.0),
                            'bbox': table.get('bbox', []),
                            'page': table.get('page', 0),
                            'source': table.get('source', 'unknown')
                        }
                        converted_tables.append(converted_table)
                        
                    elif hasattr(table_obj, 'to_dict'):  # PDFPlumber table wrapper
                        try:
                            # Try to convert to DataFrame first
                            import pandas as pd
                            df = pd.DataFrame(table_obj.to_dict('records'))
                            converted_table = {
                                'data': df.to_dict('records'),
                                'columns': df.columns.tolist(),
                                'confidence': table.get('score', 0.0),
                                'bbox': table.get('bbox', []),
                                'page': table.get('page', 0),
                                'source': table.get('source', 'unknown')
                            }
                            converted_tables.append(converted_table)
                        except Exception as e:
                            self.logger.warning(f"Failed to convert table to DataFrame: {str(e)}")
                            continue
                            
                    else:
                        self.logger.warning(f"Unknown table object type: {type(table_obj)}")
                        continue
                        
                else:
                    self.logger.warning(f"Unknown table format: {type(table)}")
                    continue
                    
            except Exception as e:
                self.logger.error(f"Failed to convert table: {str(e)}")
                continue
                
        return converted_tables



