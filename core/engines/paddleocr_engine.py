# core/engines/paddleocr_engine.py
"""
PaddleOCR引擎

封装PaddleOCR的OCR和PP-Structure表格检测功能
支持文本识别、表格检测和结构识别
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
from core.engines.base import BaseOCREngine, BaseDetectionEngine
from core.utils.logger import AppLogger


class PaddleOCREngine(BaseOCREngine, BaseDetectionEngine):
    """
    PaddleOCR引擎
    
    同时提供OCR和表格检测功能
    """
    
    def __init__(self, 
                 use_angle_cls: bool = True,
                 lang: str = 'ch',
                 use_gpu: bool = False,
                 enable_mkldnn: bool = False,
                 table_model_dir: Optional[str] = None,
                 **kwargs):
        """
        初始化PaddleOCR引擎
        
        Args:
            use_angle_cls: 是否使用角度分类器
            lang: 语言，'ch'（中文）或'en'（英文）
            use_gpu: 是否使用GPU
            enable_mkldnn: 是否启用MKLDNN加速（CPU优化）
            table_model_dir: 表格模型目录（PP-Structure）
            **kwargs: 其他参数（预留）
        """
        self.logger = AppLogger.get_logger()
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.use_gpu = use_gpu
        self.enable_mkldnn = enable_mkldnn
        self.table_model_dir = table_model_dir
        
        self._ocr = None
        self._structure_engine = None
        self._ocr_initialized = False
        self._structure_initialized = False
        self._is_ppstructure_v3 = False  # 标记是否使用 PPStructureV3
    
    @property
    def name(self) -> str:
        """引擎名称"""
        return "paddleocr"
    
    def initialize(self, **kwargs) -> bool:
        """
        初始化OCR引擎
        
        Args:
            **kwargs: 初始化参数
                - use_angle_cls: 是否使用角度分类器
                - lang: 语言
                - use_gpu: 是否使用GPU
                
        Returns:
            bool: 初始化是否成功
        """
        if self._ocr_initialized and self._ocr is not None:
            return True
        
        try:
            from paddleocr import PaddleOCR
            
            use_angle_cls = kwargs.get('use_angle_cls', self.use_angle_cls)
            lang = kwargs.get('lang', self.lang)
            use_gpu = kwargs.get('use_gpu', self.use_gpu)
            enable_mkldnn = kwargs.get('enable_mkldnn', self.enable_mkldnn)
            
            self._ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu,
                enable_mkldnn=enable_mkldnn
            )
            
            self._ocr_initialized = True
            self.logger.info(f"PaddleOCR engine initialized (lang={lang}, gpu={use_gpu})")
            return True
            
        except ImportError as e:
            self.logger.error(f"Failed to import PaddleOCR: {e}. Please install paddleocr: pip install paddleocr")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR engine: {e}")
            return False
    
    def load_models(self, **kwargs) -> bool:
        """
        加载表格检测和结构识别模型（PP-Structure）
        
        支持 PaddleOCR 3.x (PPStructureV3) 和旧版本 (PPStructure)
        
        Args:
            **kwargs: 模型配置参数
                - table_model_dir: 表格模型目录
                
        Returns:
            bool: 加载是否成功
        """
        if self._structure_initialized and self._structure_engine is not None:
            return True
        
        # 首先尝试导入 PPStructureV3 (PaddleOCR 3.x)
        try:
            from paddleocr import PPStructureV3
            
            table_model_dir = kwargs.get('table_model_dir', self.table_model_dir)
            
            # PPStructureV3 的参数可能不同，需要适配
            init_params = {
                'show_log': False,
                'image_orientation': True
            }
            
            # 如果提供了 table_model_dir，尝试使用
            if table_model_dir:
                init_params['table_model_dir'] = table_model_dir
            
            # use_gpu 参数在 PPStructureV3 中可能不存在或名称不同
            # 先尝试基本初始化
            self._structure_engine = PPStructureV3(**init_params)
            self._is_ppstructure_v3 = True
            
            self._structure_initialized = True
            self.logger.info("PaddleOCR PP-StructureV3 engine initialized")
            return True
            
        except ImportError:
            # 如果 PPStructureV3 不存在，尝试旧版本的 PPStructure
            try:
                from paddleocr import PPStructure
                
                table_model_dir = kwargs.get('table_model_dir', self.table_model_dir)
                
                init_params = {}
                if self.use_gpu:
                    init_params['use_gpu'] = self.use_gpu
                if table_model_dir:
                    init_params['table_model_dir'] = table_model_dir
                
                self._structure_engine = PPStructure(**init_params)
                self._is_ppstructure_v3 = False
                
                self._structure_initialized = True
                self.logger.info("PaddleOCR PP-Structure engine initialized (legacy version)")
                return True
                
            except ImportError as e:
                self.logger.error(f"Failed to import PPStructure or PPStructureV3: {e}")
                self.logger.error("Please ensure paddleocr is installed: pip install paddleocr")
                return False
            except Exception as e:
                self.logger.error(f"Failed to load PP-Structure models (legacy): {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load PP-StructureV3 models: {e}")
            # 如果 PPStructureV3 初始化失败，尝试回退到旧版本
            try:
                from paddleocr import PPStructure
                
                table_model_dir = kwargs.get('table_model_dir', self.table_model_dir)
                
                init_params = {}
                if self.use_gpu:
                    init_params['use_gpu'] = self.use_gpu
                if table_model_dir:
                    init_params['table_model_dir'] = table_model_dir
                
                self._structure_engine = PPStructure(**init_params)
                self._is_ppstructure_v3 = False
                
                self._structure_initialized = True
                self.logger.info("PaddleOCR PP-Structure engine initialized (fallback to legacy)")
                return True
                
            except Exception as fallback_error:
                self.logger.error(f"Failed to fallback to PPStructure: {fallback_error}")
                return False
    
    def recognize_text(self, image: Image.Image, **kwargs) -> List[Dict]:
        """
        识别文本
        
        Args:
            image: PIL Image对象
            **kwargs: 其他参数
                - det: 是否进行文本检测（默认True）
                - rec: 是否进行文本识别（默认True）
                - cls: 是否进行角度分类（默认True）
                
        Returns:
            List[Dict]: OCR结果列表，每个元素包含：
                - text: 文本内容
                - bbox: 边界框 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                - confidence: 置信度 (0-1)
        """
        if not self._ocr_initialized:
            if not self.initialize():
                return []
        
        if self._ocr is None:
            self.logger.error("PaddleOCR OCR is not available")
            return []
        
        try:
            # 转换PIL Image为numpy array
            img_array = np.array(image)
            
            # 执行OCR
            det = kwargs.get('det', True)
            rec = kwargs.get('rec', True)
            cls = kwargs.get('cls', self.use_angle_cls)
            
            ocr_results = self._ocr.ocr(img_array, det=det, rec=rec, cls=cls)
            
            # 转换结果格式
            results = []
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    if line:
                        bbox_points = line[0]  # 四个角点坐标
                        text_info = line[1]    # (文本, 置信度)
                        
                        if text_info:
                            text = text_info[0]
                            confidence = text_info[1] if len(text_info) > 1 else 1.0
                            
                            # 转换bbox格式
                            x_coords = [point[0] for point in bbox_points]
                            y_coords = [point[1] for point in bbox_points]
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            
                            results.append({
                                'text': text,
                                'bbox': bbox_points,  # 保留原始角点格式
                                'bbox_rect': [x1, y1, x2, y2],  # 添加矩形格式
                                'confidence': float(confidence)
                            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"PaddleOCR recognition failed: {e}")
            return []
    
    def detect_tables(self, image: Image.Image, **kwargs) -> List[Dict]:
        """
        检测表格
        
        Args:
            image: PIL Image对象
            **kwargs: 其他参数（保留用于未来扩展）
                
        Returns:
            List[Dict]: 检测结果列表，每个元素包含：
                - bbox: 边界框 [x1, y1, x2, y2]
                - confidence: 置信度 (0-1)
                - type: 类型（如'table'）
        """
        if not self._structure_initialized:
            if not self.load_models():
                return []
        
        if self._structure_engine is None:
            self.logger.error("PaddleOCR PP-Structure is not available")
            return []
        
        try:
            # 转换PIL Image为numpy array
            img_array = np.array(image)
            
            # 根据版本使用不同的API
            if self._is_ppstructure_v3:
                # PPStructureV3 使用 predict() 方法
                # 需要保存图像到临时文件或使用其他方式
                import tempfile
                import os
                from PIL import Image as PILImage
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    image.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                try:
                    # PPStructureV3 使用 predict() 方法，接受文件路径
                    results = self._structure_engine.predict(tmp_path)
                    
                    # 转换 PPStructureV3 结果格式
                    detection_results = []
                    for result in results:
                        # PPStructureV3 返回的结果对象，需要提取表格信息
                        # 结果可能包含多个元素，需要查找表格类型
                        if hasattr(result, 'type') and result.type == 'table':
                            # 提取边界框（PPStructureV3 的格式可能不同）
                            bbox = getattr(result, 'bbox', None)
                            if bbox:
                                if isinstance(bbox[0], (list, tuple)):
                                    x_coords = [point[0] for point in bbox]
                                    y_coords = [point[1] for point in bbox]
                                    x1, y1 = min(x_coords), min(y_coords)
                                    x2, y2 = max(x_coords), max(y_coords)
                                    bbox_rect = [x1, y1, x2, y2]
                                else:
                                    bbox_rect = bbox[:4] if len(bbox) >= 4 else bbox
                                
                                detection_results.append({
                                    'bbox': bbox_rect,
                                    'confidence': getattr(result, 'score', 1.0),
                                    'type': 'table',
                                    'raw': result  # 保留原始结果对象
                                })
                finally:
                    # 清理临时文件
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                return detection_results
            else:
                # 旧版本 PPStructure 使用 __call__ 方法
                # 注意：PPStructure的__call__方法不接受layout参数，直接传入图像数组即可
                structure_results = self._structure_engine(img_array)
                
                # 转换结果格式
                detection_results = []
                for item in structure_results:
                    if item.get('type') == 'table':
                        bbox = item.get('bbox', [])
                        if bbox and len(bbox) >= 4:
                            # bbox格式可能是[x1, y1, x2, y2]或[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            if isinstance(bbox[0], (list, tuple)):
                                # 角点格式，转换为矩形
                                x_coords = [point[0] for point in bbox]
                                y_coords = [point[1] for point in bbox]
                                x1, y1 = min(x_coords), min(y_coords)
                                x2, y2 = max(x_coords), max(y_coords)
                                bbox_rect = [x1, y1, x2, y2]
                            else:
                                # 已经是矩形格式
                                bbox_rect = bbox[:4]
                            
                            detection_results.append({
                                'bbox': bbox_rect,
                                'confidence': item.get('score', 1.0),
                                'type': 'table',
                                'raw': item  # 保留原始数据用于高级处理
                            })
                
                return detection_results
            
        except Exception as e:
            self.logger.error(f"PaddleOCR table detection failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return []
    
    def recognize_structure(self, image: Image.Image, table_bbox: Optional[List] = None, **kwargs) -> Dict:
        """
        识别表格结构
        
        Args:
            image: PIL Image对象（表格区域）
            table_bbox: 表格边界框 [x1, y1, x2, y2]（可选）
            **kwargs: 其他参数
                - return_raw: 是否返回原始输出（默认False）
                
        Returns:
            Dict: 结构识别结果，包含：
                - cells: 单元格列表
                - rows: 行信息
                - columns: 列信息
                - html: HTML格式的表格（如果可用）
                - 其他结构信息
        """
        if not self._structure_initialized:
            if not self.load_models():
                return {}
        
        if self._structure_engine is None:
            self.logger.error("PaddleOCR PP-Structure is not available")
            return {}
        
        try:
            # 如果指定了table_bbox，裁剪图像
            if table_bbox:
                x1, y1, x2, y2 = table_bbox
                image = image.crop((x1, y1, x2, y2))
            
            # 根据版本使用不同的API
            if self._is_ppstructure_v3:
                # PPStructureV3 使用 predict() 方法
                import tempfile
                import os
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    image.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                try:
                    # PPStructureV3 使用 predict() 方法
                    results = self._structure_engine.predict(tmp_path)
                    
                    # 查找表格结果
                    table_result = None
                    for result in results:
                        if hasattr(result, 'type') and result.type == 'table':
                            table_result = result
                            break
                    
                    if not table_result:
                        self.logger.warning("No table structure found in image")
                        return {}
                    
                    # 转换 PPStructureV3 结果格式
                    # PPStructureV3 的结果对象可能有不同的属性
                    result = {
                        'html': getattr(table_result, 'html', '') or getattr(table_result, 'markdown', ''),
                        'cells': getattr(table_result, 'cells', []),
                        'raw': table_result if kwargs.get('return_raw', False) else None
                    }
                    
                    # 尝试从 markdown 或 HTML 中提取信息
                    if hasattr(table_result, 'save_to_markdown'):
                        # 可以尝试获取 markdown 内容
                        import io
                        md_buffer = io.StringIO()
                        try:
                            table_result.save_to_markdown(save_path=None)  # 可能需要调整
                            result['markdown'] = md_buffer.getvalue()
                        except:
                            pass
                    
                finally:
                    # 清理临时文件
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            else:
                # 旧版本 PPStructure 使用 __call__ 方法
                # 转换PIL Image为numpy array
                img_array = np.array(image)
                
                # 执行结构识别
                structure_results = self._structure_engine(img_array)
                
                # 查找表格结果
                table_result = None
                for item in structure_results:
                    if item.get('type') == 'table':
                        table_result = item
                        break
                
                if not table_result:
                    self.logger.warning("No table structure found in image")
                    return {}
                
                # 转换结果格式
                result = {
                    'html': table_result.get('res', {}).get('html', ''),
                    'cells': table_result.get('res', {}).get('cells', []),
                    'raw': table_result if kwargs.get('return_raw', False) else None
                }
            
            # 尝试从HTML或cells中提取行列信息
            if result.get('html'):
                # 可以进一步解析HTML获取行列信息
                result['has_structure'] = True
            elif result.get('cells'):
                # 从cells中提取行列信息
                rows = set()
                cols = set()
                for cell in result['cells']:
                    if isinstance(cell, dict):
                        if 'row' in cell:
                            rows.add(cell['row'])
                        if 'col' in cell:
                            cols.add(cell['col'])
                result['rows'] = len(rows) if rows else 0
                result['columns'] = len(cols) if cols else 0
            
            return result
            
        except Exception as e:
            self.logger.error(f"PaddleOCR structure recognition failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {}
    
    def recognize_text_in_region(self, image: Image.Image, bbox: List, **kwargs) -> List[Dict]:
        """
        识别指定区域的文本
        
        Args:
            image: PIL Image对象
            bbox: 区域边界框 [x1, y1, x2, y2]
            **kwargs: 其他参数
            
        Returns:
            List[Dict]: OCR结果列表，坐标已调整到原图坐标系
        """
        try:
            # 裁剪区域
            x1, y1, x2, y2 = bbox
            cropped = image.crop((x1, y1, x2, y2))
            
            # 识别文本
            results = self.recognize_text(cropped, **kwargs)
            
            # 调整bbox坐标到原图坐标系
            for result in results:
                if 'bbox' in result:
                    # 调整角点坐标
                    adjusted_bbox = []
                    for point in result['bbox']:
                        adjusted_bbox.append([point[0] + x1, point[1] + y1])
                    result['bbox'] = adjusted_bbox
                
                if 'bbox_rect' in result:
                    # 调整矩形坐标
                    rect = result['bbox_rect']
                    result['bbox_rect'] = [rect[0] + x1, rect[1] + y1, rect[2] + x1, rect[3] + y1]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to recognize text in region: {e}")
            return []
    
    def is_available(self) -> bool:
        """
        检查引擎是否可用
        
        Returns:
            bool: 引擎是否可用
        """
        try:
            import paddleocr
            return True
        except ImportError:
            return False
