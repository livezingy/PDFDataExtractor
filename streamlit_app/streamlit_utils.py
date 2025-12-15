# streamlit_app/streamlit_utils.py
"""
Streamlit工具函数
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import pdfplumber
import sys
import asyncio

# 修复Streamlit中的asyncio事件循环问题
# Streamlit在运行时可能没有事件循环，需要在导入某些模块前处理
def setup_asyncio_for_streamlit():
    """设置asyncio事件循环以兼容Streamlit环境"""
    try:
        # 尝试获取当前事件循环
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 如果没有运行的事件循环，尝试设置一个
        try:
            # 检查是否已有事件循环策略
            if asyncio.get_event_loop_policy() is None:
                # 设置默认策略
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception as e:
            # 如果设置失败，忽略错误（某些环境下可能不需要）
            pass

# 在导入可能触发asyncio检查的模块前设置事件循环
setup_asyncio_for_streamlit()

# 延迟导入，避免在Streamlit中触发asyncio事件循环问题
try:
    from core.processing.table_processor import TableProcessor
    from core.utils.logger import AppLogger
except RuntimeError as e:
    # 如果导入时仍然出现asyncio错误，尝试再次设置
    if "no running event loop" in str(e).lower():
        setup_asyncio_for_streamlit()
        from core.processing.table_processor import TableProcessor
        from core.utils.logger import AppLogger
    else:
        raise

# 导入logging模块
import logging

# 为图片处理引入模型与解析器（按需）
try:
    from core.models.table_parser import TableParser
    from core.utils.config import Config as AppConfig
except Exception:
    TableParser = None
    AppConfig = None

# 文件大小限制（10MB，用于测试小文件）
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # 转换为字节

logger = AppLogger.get_logger()

def save_uploaded_file(uploaded_file) -> str:
    """
    保存上传的文件到临时目录
    
    Args:
        uploaded_file: Streamlit上传的文件对象
        
    Returns:
        str: 临时文件路径
    """
    # 创建临时目录
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    # 保存文件
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    logger.info(f"[Streamlit] 保存上传文件到: {temp_file_path}")
    return temp_file_path

def cleanup_temp_file(file_path: str):
    """
    清理临时文件
    
    Args:
        file_path: 临时文件路径
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[Streamlit] 清理临时文件: {file_path}")
    except Exception as e:
        logger.error(f"[Streamlit] 清理临时文件失败: {e}")

class StreamlitLogHandler(logging.Handler):
    """自定义日志处理器，用于捕获日志信息并在Streamlit中显示"""
    def __init__(self, log_messages_list):
        super().__init__()
        self.log_messages = log_messages_list
        self.setLevel(logging.INFO)
    
    def emit(self, record):
        """捕获日志记录"""
        try:
            msg = self.format(record)
            # 只捕获INFO级别的日志
            if record.levelno >= logging.INFO:
                self.log_messages.append({
                    'level': record.levelname,
                    'message': msg,
                    'time': record.created
                })
        except Exception:
            pass

def process_pdf_streamlit(
    pdf_path: str,
    method: str,
    flavor: str,
    params: Dict[str, Any],
    param_config: dict = None
) -> Dict[str, Any]:
    """
    统一的PDF处理接口（供Streamlit使用）
    
    Args:
        pdf_path: PDF文件路径
        method: 提取方法 ('pdfplumber' 或 'camelot')
        flavor: 提取flavor ('auto', 'lines', 'text', 'lattice', 'stream')
        params: 其他参数（评分阈值等）
        
    Returns:
        dict: 处理结果，包含：
            - file_info: 文件基本信息
            - detection_steps: 检测过程信息
            - tables: 提取的表格列表
            - visualizations: 可视化结果
            - log_messages: 日志信息列表
    """
    # 用于捕获日志信息
    log_messages = []
    streamlit_handler = StreamlitLogHandler(log_messages)
    
    # 获取根logger并添加处理器
    root_logger = logging.getLogger('PDFExtractor')
    root_logger.addHandler(streamlit_handler)
    
    try:
        results = {
            'file_info': {},
            'detection_steps': [],
            'extracted_tables': [],
            'visualizations': {},
            'log_messages': log_messages
        }
        
        # 1. 获取文件基本信息
        file_size = os.path.getsize(pdf_path)
        results['file_info'] = {
            'file_name': os.path.basename(pdf_path),
            'file_size': file_size,
            'file_size_mb': file_size / 1024 / 1024,
            'file_path': pdf_path
        }
        
        # 2. 更新参数
        params['current_filepath'] = pdf_path
        params['table_method'] = method
        params['table_flavor'] = flavor
        
        # 3. 根据文件类型分支处理
        ext = os.path.splitext(pdf_path)[1].lower()

        # ========= 3A. 处理图片文件：根据选择的引擎处理 =========
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            # 获取选择的检测引擎（从method参数，默认为paddleocr）
            # method可能是'PaddleOCR'或'Transformer'（首字母大写）
            method_lower = method.lower() if method else 'paddleocr'
            detection_engine = method_lower if method_lower in ['paddleocr', 'transformer'] else 'paddleocr'
            
            results['detection_steps'].append({
                'step': 2,
                'name': 'Image Detected',
                'status': 'info',
                'message': f'Image file detected. Using {detection_engine.upper()} engine'
            })

            try:
                from PIL import Image
                image = Image.open(pdf_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # 根据选择的引擎处理
                if detection_engine == 'paddleocr':
                    extracted = _process_image_with_paddleocr(image, params, results)
                else:  # transformer
                    extracted = _process_image_with_transformer(image, params, results)

                # 转换为Streamlit显示格式
                results['extracted_tables'] = format_tables_for_streamlit(extracted)

                # 记录参数信息
                results['extraction_params'] = {
                    'method': detection_engine,
                    'flavor': None
                }

                return results
            except Exception as e:
                logger.error(f"[Streamlit] Image processing failed: {e}", exc_info=True)
                results['detection_steps'].append({
                    'step': 3,
                    'name': f'{detection_engine.upper()} Processing',
                    'status': 'error',
                    'message': f'Processing failed: {str(e)}'
                })
                return results

        # ========= 3B. 处理PDF文件（原流程） =========
        with pdfplumber.open(pdf_path) as pdf:
            results['file_info']['total_pages'] = len(pdf.pages)
            
            # 添加检测步骤
            results['detection_steps'].append({
                'step': 2,
                'name': 'File Loading',
                'status': 'success',
                'message': f'Successfully loaded PDF file with {len(pdf.pages)} page(s)'
            })
            
            # 4. 使用TableProcessor处理每一页
            all_tables = []
            table_processor = TableProcessor(params)
            
            step_counter = 1  # 步骤计数器，从1开始（步骤0是文件加载）
            
            for page_num, page in enumerate(pdf.pages, 1):
                step_counter += 1
                # 添加处理中步骤
                processing_step = {
                    'step': step_counter,
                    'name': f'Processing Page {page_num}',
                    'status': 'processing',
                    'message': f'Analyzing page features...'
                }
                results['detection_steps'].append(processing_step)
                
                try:
                    # 处理页面
                    # 确保flavor参数正确传递
                    # TableProcessor中，如果flavor是None，会自动判断
                    # 如果flavor是'auto'，需要转换为None
                    if flavor == 'auto':
                        params['table_flavor'] = None  # TableProcessor会自动判断
                    else:
                        params['table_flavor'] = flavor
                    
                    # Add parameter configuration to params
                    if param_config:
                        method_lower = method.lower()
                        flavor_lower = flavor.lower() if flavor != 'auto' else None
                        
                        if method_lower == 'pdfplumber':
                            params['pdfplumber_param_mode'] = param_config.get('mode', 'auto')
                            if param_config.get('mode') == 'custom':
                                params['pdfplumber_custom_params'] = param_config.get('params', {})
                        elif method_lower == 'camelot' and flavor_lower:
                            if flavor_lower == 'lattice':
                                params['camelot_lattice_param_mode'] = param_config.get('mode', 'auto')
                                if param_config.get('mode') == 'custom':
                                    params['camelot_lattice_custom_params'] = param_config.get('params', {})
                            elif flavor_lower == 'stream':
                                params['camelot_stream_param_mode'] = param_config.get('mode', 'auto')
                                if param_config.get('mode') == 'custom':
                                    params['camelot_stream_custom_params'] = param_config.get('params', {})
                    
                    # 重新创建TableProcessor以确保参数更新
                    table_processor = TableProcessor(params)
                    
                    # 获取参数信息（用于显示）
                    # 使用TableProcessor中的PageFeatureAnalyzer（适配器类，有predict_table_type方法）
                    from core.processing.table_processor import PageFeatureAnalyzer
                    feature_analyzer = PageFeatureAnalyzer(page)
                    
                    # 收集参数信息
                    extraction_params = {
                        'method': method,
                        'flavor': flavor if flavor != 'auto' else None
                    }
                    
                    # 根据方法和flavor获取参数
                    if method == 'pdfplumber':
                        table_type = feature_analyzer.predict_table_type()
                        actual_flavor = flavor if flavor != 'auto' else ('lines' if table_type == 'bordered' else 'text')
                        extraction_params['flavor'] = actual_flavor
                        
                        # 获取默认参数
                        default_params = {
                            'snap_tolerance': 2,
                            'join_tolerance': 2,
                            'edge_min_length': 3,
                            'intersection_tolerance': 3,
                            'min_words_vertical': 1,
                            'min_words_horizontal': 1,
                            'text_x_tolerance': 3,
                            'text_y_tolerance': 5
                        }
                        
                        # 获取计算后的参数
                        calculated_params = feature_analyzer.get_pdfplumber_params(table_type)
                        
                        extraction_params['default_params'] = default_params
                        extraction_params['calculated_params'] = calculated_params
                        results['extraction_params'] = extraction_params
                    
                    elif method == 'camelot':
                        table_type = feature_analyzer.predict_table_type()
                        actual_flavor = flavor if flavor != 'auto' else ('lattice' if table_type == 'bordered' else 'stream')
                        extraction_params['flavor'] = actual_flavor
                        
                        if actual_flavor == 'lattice':
                            # 获取默认参数
                            default_params = {
                                'flavor': 'lattice',
                                'line_scale': 40,
                                'line_tol': 2,
                                'joint_tol': 2
                            }
                            
                            # 获取计算后的参数
                            image_shape = (int(page.height * 2), int(page.width * 2))
                            calculated_params = feature_analyzer.get_camelot_lattice_params(image_shape)
                            
                            extraction_params['default_params'] = default_params
                            extraction_params['calculated_params'] = calculated_params
                            results['extraction_params'] = extraction_params
                        
                        elif actual_flavor == 'stream':
                            # 获取默认参数
                            default_params = {
                                'flavor': 'stream',
                                'edge_tol': 50,
                                'row_tol': 2,
                                'column_tol': 0
                            }
                            
                            # 获取计算后的参数
                            calculated_params = feature_analyzer.get_camelot_stream_params()
                            
                            extraction_params['default_params'] = default_params
                            extraction_params['calculated_params'] = calculated_params
                            results['extraction_params'] = extraction_params
                    
                    page_tables = table_processor.process_pdf_page(pdf_path, page)
                    
                    # 添加页面号到每个表格
                    for table in page_tables:
                        if isinstance(table, dict):
                            table['page_num'] = page_num
                    
                    # 更新处理中步骤为完成状态
                    step_counter += 1
                    # 移除之前的processing步骤
                    results['detection_steps'] = [s for s in results['detection_steps'] if s != processing_step]
                    
                    # 检查返回的表格是否有效
                    if page_tables and isinstance(page_tables, list) and len(page_tables) > 0:
                        # 检查表格数据是否有效
                        valid_tables = []
                        for table in page_tables:
                            # 检查表格是否有有效的data字段
                            if isinstance(table, dict):
                                table_data = table.get('data')
                                if table_data is not None:
                                    # 如果是DataFrame，检查是否为空
                                    try:
                                        import pandas as pd
                                        if isinstance(table_data, pd.DataFrame):
                                            if not table_data.empty:
                                                valid_tables.append(table)
                                        else:
                                            valid_tables.append(table)
                                    except:
                                        valid_tables.append(table)
                                else:
                                    # 检查table对象是否有df属性
                                    table_obj = table.get('table')
                                    if table_obj is not None:
                                        if hasattr(table_obj, 'df'):
                                            import pandas as pd
                                            if isinstance(table_obj.df, pd.DataFrame) and not table_obj.df.empty:
                                                valid_tables.append(table)
                                        else:
                                            valid_tables.append(table)
                                    else:
                                        valid_tables.append(table)
                        
                        if valid_tables:
                            all_tables.extend(valid_tables)
                            results['detection_steps'].append({
                                'step': step_counter,
                                'name': f'Processing Page {page_num}',
                                'status': 'success',
                                'message': f'Successfully extracted {len(valid_tables)} table(s)'
                            })
                        else:
                            results['detection_steps'].append({
                                'step': step_counter,
                                'name': f'Processing Page {page_num}',
                                'status': 'success',
                                'message': f'No valid tables detected'
                            })
                    else:
                        results['detection_steps'].append({
                            'step': step_counter,
                            'name': f'Processing Page {page_num}',
                            'status': 'success',
                            'message': f'No tables detected'
                        })
                    
                except Exception as e:
                    step_counter += 1
                    error_msg = str(e)
                    logger.error(f"[Streamlit] Processing page {page_num} failed: {error_msg}", exc_info=True)
                    # 移除processing步骤
                    results['detection_steps'] = [s for s in results['detection_steps'] if s != processing_step]
                    results['detection_steps'].append({
                        'step': step_counter,
                        'name': f'Processing Page {page_num}',
                        'status': 'error',
                        'message': f'Processing failed: {error_msg}'
                    })
        
            # 5. 格式化表格结果
            results['extracted_tables'] = format_tables_for_streamlit(all_tables)
            
            # 6. 添加最终统计
            step_counter += 1
            results['detection_steps'].append({
                'step': step_counter,
                'name': 'Processing Complete',
                'status': 'success',
                'message': f'Total extracted: {len(all_tables)} table(s)'
            })
        
    except Exception as e:
        logger.error(f"[Streamlit] PDF processing failed: {e}")
        step_counter += 1
        results['detection_steps'].append({
            'step': step_counter,
            'name': 'Processing Failed',
            'status': 'error',
            'message': str(e)
        })
        raise
    finally:
        # 移除日志处理器
        root_logger.removeHandler(streamlit_handler)
    
    return results

def format_tables_for_streamlit(tables: List[Dict]) -> List[Dict]:
    """
    格式化表格结果供Streamlit显示
    
    Args:
        tables: 原始表格列表（来自TableProcessor或PaddleOCR/Transformer）
        
    Returns:
        list: 格式化后的表格列表
    """
    formatted_tables = []
    
    for i, table in enumerate(tables):
        try:
            import pandas as pd
            
            # 获取表格数据
            # TableProcessor返回的格式：
            # {'table': wrapper/table_obj, 'bbox': ..., 'score': ..., 'source': ..., ...}
            # PaddleOCR返回的格式：
            # {'df': DataFrame, 'bbox': ..., 'score': ..., 'method': 'paddleocr', ...}
            # Transformer返回的格式：
            # {'table': DataFrame, 'bbox': ..., 'score': ..., ...}
            
            table_obj = table.get('table')
            table_data = None
            rows = 0
            cols = 0
            
            # 首先检查是否有直接的df键（PaddleOCR格式）
            if 'df' in table and table['df'] is not None:
                table_data = table['df']
                if isinstance(table_data, pd.DataFrame):
                    rows = table_data.shape[0] if not table_data.empty else 0
                    cols = table_data.shape[1] if not table_data.empty else 0
            # 处理不同的表格对象类型
            elif table_obj is not None:
                # 如果是PDFPlumberTableWrapper，数据在df属性中
                if hasattr(table_obj, 'df'):
                    table_data = table_obj.df
                    rows = table_data.shape[0] if not table_data.empty else 0
                    cols = table_data.shape[1] if not table_data.empty else 0
                # 如果是Camelot Table对象，使用df属性
                elif hasattr(table_obj, 'df'):
                    table_data = table_obj.df
                    rows = table_data.shape[0] if not table_data.empty else 0
                    cols = table_data.shape[1] if not table_data.empty else 0
                # 如果是DataFrame
                elif isinstance(table_obj, pd.DataFrame):
                    table_data = table_obj
                    rows = table_data.shape[0] if not table_data.empty else 0
                    cols = table_data.shape[1] if not table_data.empty else 0
                # 尝试extract方法（pdfplumber Table对象）
                elif hasattr(table_obj, 'extract'):
                    try:
                        extracted = table_obj.extract()
                        if extracted:
                            table_data = pd.DataFrame(extracted)
                            rows = table_data.shape[0] if not table_data.empty else 0
                            cols = table_data.shape[1] if not table_data.empty else 0
                    except:
                        pass
            
            # 如果没有获取到数据，尝试从table字典中获取
            if table_data is None:
                table_data = table.get('data')
                if table_data is not None and isinstance(table_data, pd.DataFrame):
                    rows = table_data.shape[0] if not table_data.empty else 0
                    cols = table_data.shape[1] if not table_data.empty else 0
            
            # 获取method和flavor信息
            # 优先使用table字典中的method字段（PaddleOCR/Transformer格式）
            method = table.get('method', 'unknown')
            flavor = table.get('flavor', 'unknown')
            
            # 如果没有method字段，从source解析（PDFPlumber/Camelot格式）
            if method == 'unknown':
                source = table.get('source', 'unknown')
                if 'pdfplumber' in source:
                    method = 'pdfplumber'
                    if 'lines' in source:
                        flavor = 'lines'
                    elif 'text' in source:
                        flavor = 'text'
                elif 'camelot' in source:
                    method = 'camelot'
                    if 'lattice' in source:
                        flavor = 'lattice'
                    elif 'stream' in source:
                        flavor = 'stream'
                elif 'paddleocr' in source.lower() or method == 'paddleocr':
                    method = 'paddleocr'
                    flavor = None
                elif 'transformer' in source.lower() or method == 'transformer':
                    method = 'transformer'
                    flavor = None
            
            # 获取页面号（从table对象或bbox推断）
            page_num = table.get('page_num', 0)
            if page_num == 0 and hasattr(table_obj, 'page'):
                page = table_obj.page
                if hasattr(page, 'page_number'):
                    page_num = page.page_number
            
            formatted_table = {
                'id': i + 1,
                'data': table_data,
                'score': table.get('score', 0.0),
                'method': method,
                'flavor': flavor,
                'page_num': page_num,
                'bbox': table.get('bbox'),
                'rows': rows,
                'cols': cols
            }
            formatted_tables.append(formatted_table)
        except Exception as e:
            logger.error(f"[Streamlit] 格式化表格 {i+1} 失败: {e}", exc_info=True)
            continue
    
    return formatted_tables

def _process_image_with_paddleocr(image, params: Dict[str, Any], results: Dict[str, Any]) -> List[Dict]:
    """
    使用PaddleOCR处理图像文件
    
    Args:
        image: PIL Image对象
        params: 处理参数
        results: 结果字典（用于更新检测步骤）
        
    Returns:
        List[Dict]: 提取的表格列表
    """
    try:
        from core.engines.factory import EngineFactory
        
        # 检查模型是否已预加载
        models_ready = False
        try:
            from core.utils.paddleocr_model_preloader import check_paddleocr_models_exist
            models_ready = check_paddleocr_models_exist()
        except ImportError:
            pass
        
        if models_ready:
            results['detection_steps'].append({
                'step': 3,
                'name': 'PaddleOCR Initialization',
                'status': 'info',
                'message': 'Initializing PaddleOCR engine... Models are ready.'
            })
        else:
            results['detection_steps'].append({
                'step': 3,
                'name': 'PaddleOCR Initialization',
                'status': 'info',
                'message': 'Initializing PaddleOCR engine... This may take 2-5 minutes on first use as models are being downloaded. Please be patient.'
            })
        
        # 创建PaddleOCR检测引擎
        detection_engine = EngineFactory.create_detection('paddleocr', use_gpu=False)
        
        # 加载模型
        # 注意：首次使用时，PPStructureV3 会下载多个模型文件（约200-500MB），
        # 这可能需要2-5分钟，特别是在Streamlit Cloud环境中。
        # 如果下载过程中出现超时，请稍后重试或使用其他提取方法（如PDFPlumber或Camelot）
        try:
            if not detection_engine.load_models():
                results['detection_steps'].append({
                    'step': 3,
                    'name': 'PaddleOCR Initialization',
                    'status': 'error',
                    'message': 'Model loading failed. Please try again or use a different extraction method (PDFPlumber or Camelot).'
                })
                raise RuntimeError("Failed to load PaddleOCR models")
        except RuntimeError as e:
            # 检查是否是依赖错误
            error_msg = str(e)
            if "dependency" in error_msg.lower() or "paddlex" in error_msg.lower():
                results['detection_steps'].append({
                    'step': 3,
                    'name': 'PaddleOCR Initialization',
                    'status': 'error',
                    'message': 'PaddleOCR dependency error. The system is trying to fall back to legacy version...'
                })
                # 错误已经在引擎内部处理，这里只是记录
                raise
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                results['detection_steps'].append({
                    'step': 3,
                    'name': 'PaddleOCR Initialization',
                    'status': 'error',
                    'message': 'Model loading timeout. This usually happens on first use when downloading models. Please wait a few minutes and try again, or use PDFPlumber/Camelot instead.'
                })
                raise RuntimeError(
                    "PaddleOCR model loading timeout. "
                    "This usually happens on first use when models are being downloaded (may take 2-5 minutes). "
                    "Please wait a few minutes and try again, or use a different extraction method (PDFPlumber or Camelot)."
                ) from e
            else:
                raise
        
        results['detection_steps'].append({
            'step': 4,
            'name': 'PaddleOCR Table Detection',
            'status': 'info',
            'message': 'Detecting tables in image...'
        })
        
        # 检测表格
        tables = detection_engine.detect_tables(image)
        
        if not tables:
            results['detection_steps'].append({
                'step': 5,
                'name': 'PaddleOCR Processing',
                'status': 'warning',
                'message': 'No tables detected in image'
            })
            return []
        
        results['detection_steps'].append({
            'step': 5,
            'name': 'PaddleOCR Table Detection',
            'status': 'success',
            'message': f'Detected {len(tables)} table(s)'
        })
        
        # 识别每个表格的结构
        extracted_tables = []
        base_step = 6
        for idx, table in enumerate(tables):
            bbox = table['bbox']
            confidence = table['confidence']
            
            # 裁剪表格区域
            table_image = image.crop(bbox)
            
            results['detection_steps'].append({
                'step': base_step + idx * 2,
                'name': f'PaddleOCR Structure Recognition (Table {idx+1})',
                'status': 'info',
                'message': f'Recognizing structure for table {idx+1}...'
            })
            
            # 识别结构
            structure = detection_engine.recognize_structure(table_image, return_raw=True)
            
            # 如果有HTML输出，尝试解析为DataFrame
            import pandas as pd
            df = pd.DataFrame()
            
            if structure.get('html'):
                try:
                    from io import StringIO
                    # 尝试从HTML读取表格
                    html_tables = pd.read_html(StringIO(structure['html']))
                    if html_tables and len(html_tables) > 0:
                        df = html_tables[0]
                        logger.info(f"Successfully parsed HTML table with shape: {df.shape}")
                except Exception as e:
                    logger.warning(f"Failed to parse HTML table: {e}")
                    # 如果解析失败，尝试从cell_bbox构建DataFrame（如果可用）
                    # 注意：PaddleOCR的res中可能包含cell_bbox而不是cells
                    if structure.get('raw') and structure['raw'].get('res'):
                        res = structure['raw']['res']
                        if 'cell_bbox' in res:
                            # 可以尝试从cell_bbox和OCR结果构建表格
                            # 这里暂时创建空DataFrame，后续可以增强
                            logger.debug("cell_bbox available but HTML parsing failed")
            
            # 转换为标准格式（与TableProcessor返回格式兼容）
            table_data = {
                'table': df,  # 使用'table'键以兼容format_tables_for_streamlit
                'df': df,     # 同时提供'df'键
                'bbox': bbox,
                'score': confidence,
                'source': 'paddleocr',
                'method': 'paddleocr',
                'flavor': None,
                'structure': structure,
                'page_num': 1  # 图像文件默认为第1页
            }
            
            extracted_tables.append(table_data)
            
            results['detection_steps'].append({
                'step': base_step + idx * 2 + 1,
                'name': f'PaddleOCR Structure Recognition (Table {idx+1})',
                'status': 'success',
                'message': f'Table {idx+1} structure recognized (shape: {df.shape if not df.empty else "empty"})'
            })
        
        final_step = base_step + len(tables) * 2
        results['detection_steps'].append({
            'step': final_step,
            'name': 'PaddleOCR Processing',
            'status': 'success',
            'message': f'PaddleOCR processed {len(extracted_tables)} table(s) successfully'
        })
        
        return extracted_tables
        
    except ImportError as e:
        logger.error(f"PaddleOCR not available: {e}")
        results['detection_steps'].append({
            'step': 3,
            'name': 'PaddleOCR Processing',
            'status': 'error',
            'message': f'PaddleOCR not available: {str(e)}. Please install: pip install paddleocr paddlepaddle'
        })
        return []
    except Exception as e:
        logger.error(f"PaddleOCR processing failed: {e}", exc_info=True)
        results['detection_steps'].append({
            'step': 3,
            'name': 'PaddleOCR Processing',
            'status': 'error',
            'message': f'Processing failed: {str(e)}'
        })
        return []


def _process_image_with_transformer(image, params: Dict[str, Any], results: Dict[str, Any]) -> List[Dict]:
    """
    使用Transformer处理图像文件（仅本地环境）
    
    Args:
        image: PIL Image对象
        params: 处理参数
        results: 结果字典（用于更新检测步骤）
        
    Returns:
        List[Dict]: 提取的表格列表
    """
    # 检测是否在Streamlit Cloud环境
    is_streamlit_cloud = os.environ.get('STREAMLIT_CLOUD', '').lower() == 'true' or \
                        'STREAMLIT_SHARING' in os.environ or \
                        os.path.exists('/home/appuser')
    
    if is_streamlit_cloud:
        error_msg = "Transformer is not available in Streamlit Cloud. Please use PaddleOCR or deploy locally."
        logger.error(error_msg)
        results['detection_steps'].append({
            'step': 3,
            'name': 'Transformer Processing',
            'status': 'error',
            'message': error_msg + ' See deployment guide for local setup.'
        })
        return []
    
    try:
        # 初始化 TableParser（与 GUI 保持一致的配置来源）
        app_cfg = AppConfig().DEFAULT_CONFIG if AppConfig else {}
        table_parser = TableParser(app_cfg) if TableParser else None
        if not table_parser:
            raise RuntimeError("TableParser not available. Transformer requires local deployment.")

        results['detection_steps'].append({
            'step': 3,
            'name': 'Transformer Initialization',
            'status': 'info',
            'message': 'Initializing Transformer models...'
        })

        # 运行检测与解析
        parsing = asyncio.run(table_parser.parser_image(image, params))

        # 规范化结果
        extracted = parsing.get('tables', []) if isinstance(parsing, dict) else []
        results['detection_steps'].append({
            'step': 4,
            'name': 'Transformer Processing',
            'status': 'success' if parsing.get('success', True) else 'error',
            'message': f"Transformer parsed {len(extracted)} table(s)"
        })

        return extracted
        
    except Exception as e:
        logger.error(f"Transformer processing failed: {e}", exc_info=True)
        results['detection_steps'].append({
            'step': 3,
            'name': 'Transformer Processing',
            'status': 'error',
            'message': f'Processing failed: {str(e)}. Transformer requires local deployment with sufficient resources.'
        })
        return []


def check_dependencies() -> Dict[str, bool]:
    """
    检查依赖是否可用
    
    Returns:
        dict: 依赖状态字典
    """
    # 检测是否在Streamlit Cloud环境
    is_streamlit_cloud = os.environ.get('STREAMLIT_CLOUD', '').lower() == 'true' or \
                        'STREAMLIT_SHARING' in os.environ or \
                        os.path.exists('/home/appuser')
    
    dependencies = {
        'pdfplumber': False,
        'camelot': False,
        'transformer': False,
        'paddleocr': False
    }
    
    # 检查pdfplumber
    try:
        import pdfplumber
        dependencies['pdfplumber'] = True
    except ImportError:
        pass
    
    # 检查camelot
    # 在导入camelot之前设置环境变量，避免在无头环境中加载OpenGL库
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('DISPLAY', '')
    try:
        import camelot
        dependencies['camelot'] = True
    except ImportError:
        pass
    except Exception as e:
        # 捕获其他可能的导入错误（如libGL.so.1）
        logger.warning(f"Camelot import failed: {e}")
        pass
    
    # 检查transformer（仅在本地环境）
    if not is_streamlit_cloud:
        try:
            from core.models.table_parser import TableParser
            dependencies['transformer'] = True
        except (ImportError, Exception):
            pass
    else:
        # Streamlit Cloud环境：Transformer不可用
        dependencies['transformer'] = False
    
    # 检查paddleocr
    try:
        from core.engines.factory import EngineFactory
        if EngineFactory.is_detection_registered('paddleocr'):
            # 尝试创建引擎实例检查是否可用
            engine = EngineFactory.create_detection('paddleocr', use_gpu=False)
            dependencies['paddleocr'] = engine.is_available()
    except (ImportError, Exception) as e:
        logger.debug(f"PaddleOCR check failed: {e}")
        pass
    
    return dependencies

