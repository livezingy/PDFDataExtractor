# core/processing/page_feature_analyzer.py
"""
页面特征分析器

职责：
- 提取PDF页面的基础特征（字符、文本行、单词、线条）
- 计算统计指标（平均宽度、高度、间距等）
- 线条合并和规范化
- Rects到Lines的转换

不负责：
- 表格类型判断
- 参数计算
"""

# from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import math
from collections import Counter
from scipy.spatial import KDTree
from core.utils.logger import AppLogger


class PageFeatureAnalyzer:
    """
    页面特征(lines, curves, rects, chars, text_lines)分析器
    """
    
    def __init__(self, page):
        """
        初始化页面特征分析器并自动执行所有分析
        
        Args:
            page: pdfplumber.Page对象
        """
        self.page = page
        self.logger = AppLogger.get_logger()
        
        # 步骤1: 收集所有基础元素
        self.lines = list(page.lines) if hasattr(page, 'lines') else []
        self.curves = page.curves if hasattr(page, 'curves') else []
        self.rects = page.rects if hasattr(page, 'rects') else []
        self.chars = page.chars if hasattr(page, 'chars') else []
        self.text_lines = page.text_lines if hasattr(page, 'text_lines') else []
        
        # 步骤2: 先分析字符特征（获取 min_height 用于后续的线条合并阈值计算）
        self._analyze_chars()
        
        
        # 步骤3: 分析其他特征
        self._analyze_text_lines()
        self._analyze_lines()        
        
        # 输出详细的页面元素调试信息
        self._log_page_elements()        
        
   
    
    @staticmethod
    def _get_mode_with_fallback(values, min_count=3):
        """
        计算众数，如果数据不足则回退到最小值
        
        Args:
            values: 数值列表
            min_count: 最小数据量要求
            
        Returns:
            float: 众数（如果数据足够），否则返回最小值
        """
        if not values:
            return 0
        
        if len(values) < min_count:
            return min(values)
        
        counter = Counter(values)
        most_common = counter.most_common(1)
        
        if most_common and most_common[0][1] >= 2:  # 至少出现2次
            return most_common[0][0]
        else:
            # 如果所有值都不同，回退到最小值
            return min(values)   
    
    
    
    def _convert_rects_to_lines(self, char_height_mode, rects):
        """
        将Rects转换为Lines
        
        根据水平线或垂线的长度与合并阈值的关系决定生成线段的条数。
        
        Rect属性说明（基于pdfplumber）：
        - 必有属性：x0, y0, x1, y1（或 top, bottom, left, right）
        - 可选属性：width, height, stroke, fill, linewidth
        - PDF坐标系：原点在左下角，y轴向上为正
        - pdfplumber坐标系：原点在左上角，y轴向下为正（已转换）
        
        Args:
            char_height_mode: 字符高度众数（来自 char_analysis，用于合并阈值）
            rects: 待转换的 rects 列表
            
        Returns:
            tuple: (h_lines, v_lines) - 分别返回垂线和水平线，不返回合并的线
        """       
        
        merge_threshold = char_height_mode / 4 if char_height_mode > 0 else 2.0
        h_lines = []
        v_lines = []
        invalid_count = 0
        for rect in rects:
            try:
                # 提取坐标 - 完善的属性获取策略
                # pdfplumber的rect通常是字典类型，支持多种访问方式
                # 优先尝试字典访问（x0/y0/x1/y1），如果不存在则尝试其他键名（left/right/top/bottom）
                # 如果rect是对象，尝试属性访问
                if isinstance(rect, dict):
                    # 字典类型：优先使用 x0/y0/x1/y1，否则使用 left/right/top/bottom
                    x0 = rect.get('x0') or rect.get('left')
                    x1 = rect.get('x1') or rect.get('right')
                    y0 = rect.get('y0') or rect.get('top')
                    y1 = rect.get('y1') or rect.get('bottom')
                else:
                    # 对象类型：尝试属性访问
                    x0 = getattr(rect, 'x0', None) or getattr(rect, 'left', None)
                    x1 = getattr(rect, 'x1', None) or getattr(rect, 'right', None)
                    y0 = getattr(rect, 'y0', None) or getattr(rect, 'top', None)
                    y1 = getattr(rect, 'y1', None) or getattr(rect, 'bottom', None)
                
                # 检查必需坐标是否完整
                if any(coord is None for coord in [x0, x1, y0, y1]):
                    invalid_count += 1
                    self.logger.debug(
                        f"Skipping rect with incomplete coordinates: "
                        f"x0={x0}, x1={x1}, y0={y0}, y1={y1}"
                    )
                    continue
                
                # 计算或获取宽高
                if isinstance(rect, dict):
                    width = rect.get('width') or abs(x1 - x0) if x0 is not None and x1 is not None else 0
                    height = rect.get('height') or abs(y1 - y0) if y0 is not None and y1 is not None else 0
                else:
                    width = getattr(rect, 'width', None) or (abs(x1 - x0) if x0 is not None and x1 is not None else 0)
                    height = getattr(rect, 'height', None) or (abs(y1 - y0) if y0 is not None and y1 is not None else 0)
                
                
                if width <= 0 or height <= 0:
                    invalid_count += 1
                    self.logger.debug(
                        f"Skipping rect with invalid width or height: "
                        f"width={width}, height={height}"
                    )
                    continue
                
                
                # 获取线宽
                if isinstance(rect, dict):
                    linewidth = rect.get('linewidth', 1)
                else:
                    linewidth = getattr(rect, 'linewidth', 1)
                
                # 根据水平线或垂线的长度与合并阈值的关系决定生成的直线的数量
                h_length = abs(x1 - x0)
                v_length = abs(y1 - y0)
                if h_length > v_length:
                    # 水平线更长
                    if v_length > merge_threshold:
                        # 生成两条水平线，两条垂线
                        h_lines.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y0, 'linewidth': linewidth})
                        h_lines.append({'x0': x0, 'y0': y1, 'x1': x1, 'y1': y1, 'linewidth': linewidth})
                        v_lines.append({'x0': x0, 'y0': y0, 'x1': x0, 'y1': y1, 'linewidth': linewidth})
                        v_lines.append({'x0': x1, 'y0': y0, 'x1': x1, 'y1': y1, 'linewidth': linewidth})
                    elif h_length > merge_threshold:
                        # 仅生成一条水平线
                        h_lines.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y0, 'linewidth': linewidth})
                else:
                    # 垂线更长或相等
                    if h_length > merge_threshold:
                        # 生成两条水平线，两条垂线
                        h_lines.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y0, 'linewidth': linewidth})
                        h_lines.append({'x0': x0, 'y0': y1, 'x1': x1, 'y1': y1, 'linewidth': linewidth})
                        v_lines.append({'x0': x0, 'y0': y0, 'x1': x0, 'y1': y1, 'linewidth': linewidth})
                        v_lines.append({'x0': x1, 'y0': y0, 'x1': x1, 'y1': y1, 'linewidth': linewidth})
                    elif v_length > merge_threshold:
                        # 仅生成一条垂线
                        v_lines.append({'x0': x0, 'y0': y0, 'x1': x0, 'y1': y1, 'linewidth': linewidth})
                    
            except (KeyError, TypeError, ValueError) as e:
                invalid_count += 1
                self.logger.debug(f"Failed to convert rect to lines: {e}")
        
        # 统计信息
        total_converted = len(h_lines) + len(v_lines)
        if total_converted == 0:
            if invalid_count > 0:
                self.logger.warning(
                    f"[Rects->Lines] 无有效线条生成，跳过了 {invalid_count} 个无效rects"
                )
                return ([], [])    
            else:
                return ([], [])

        # 分别返回垂线和水平线，不返回合并的线
        return (h_lines, v_lines)
        
        
    def _analyze_lines(self):
        """
        分析线条特征，包括方向、长度、间距和分布
        
        处理流程：
        1. 先对 page.lines 调用 _determine_line_orientation_by_linewidth 进行分类
        2. 对 page.rects 调用 _convert_rects_to_lines
        3. 分别进行长度过滤
        4. 最后统一计算 min/max/mode 值
        """
        # 初始化结果
        horizontal_lines = []
        vertical_lines = []       
        horizontal_lengths = []
        vertical_lengths = []
        all_line_widths = []
        
        # 计算最小长度要求（使用最小字符高度）
        min_length = self.char_analysis['min_height'] * 1.5 if self.char_analysis['min_height'] > 0 else 10

        # 2.1 处理 page.lines
        if self.lines:
            for line in self.lines: 
                # 计算线条长度
                length = math.sqrt((line['x1'] - line['x0'])**2 + (line['y1'] - line['y0'])**2)
                
                # 保留满足最小长度要求的线条
                if length >= min_length:
                    x0, y0, x1, y1 = line['x0'], line['y0'], line['x1'], line['y1']
                    line_width = line.get('linewidth', 0)
                    
                    # 确定方向
                    orientation = self._determine_line_orientation_by_linewidth(x0, y0, x1, y1, line_width)
                    
                    if orientation == 'horizontal':
                        horizontal_lengths.append(abs(x1 - x0))
                        if line_width > 0:
                            all_line_widths.append(line_width)
                        horizontal_lines.append(line)
                    elif orientation == 'vertical':
                        vertical_lengths.append(abs(y1 - y0))
                        if line_width > 0:
                            all_line_widths.append(line_width)
                        vertical_lines.append(line)
        
        # 2.2 处理 page.rects
        if self.rects:
            # 调用 _convert_rects_to_lines
            h_lines_from_rects, v_lines_from_rects = self._convert_rects_to_lines(
                char_height_mode=self.char_analysis['min_height'] if self.char_analysis['min_height'] > 0 else 2.0,
                rects=self.rects
            )
            
            # 对返回的垂线和水平线分别进行长度过滤
            for h_line in h_lines_from_rects:
                length = abs(h_line['x1'] - h_line['x0'])
                if length >= min_length:
                    horizontal_lengths.append(length)
                    if h_line.get('linewidth', 0) > 0:
                        all_line_widths.append(h_line['linewidth'])
                    horizontal_lines.append(h_line)
            
            for v_line in v_lines_from_rects:
                length = abs(v_line['y1'] - v_line['y0'])
                if length >= min_length:
                    vertical_lengths.append(length)
                    if v_line.get('linewidth', 0) > 0:
                        all_line_widths.append(v_line['linewidth'])
                    vertical_lines.append(v_line)
        
        # 2.3 最后对得到的所有水平线和垂直线进行计算，分别得到 min/max/mode 值
        self.line_analysis = {
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'horizontal_lines_length': horizontal_lengths,
            'vertical_lines_length': vertical_lengths,
            'line_widths': all_line_widths,
            'min_horizontal_length': np.min(horizontal_lengths) if horizontal_lengths else 0,
            'max_horizontal_length': np.max(horizontal_lengths) if horizontal_lengths else 0,
            'mode_horizontal_length': self._get_mode_with_fallback(horizontal_lengths) if horizontal_lengths else 0,
            'min_vertical_length': np.min(vertical_lengths) if vertical_lengths else 0,
            'max_vertical_length': np.max(vertical_lengths) if vertical_lengths else 0,
            'mode_vertical_length': self._get_mode_with_fallback(vertical_lengths) if vertical_lengths else 0,
            'min_line_width': np.min(all_line_widths) if all_line_widths else 0,
            'max_line_width': np.max(all_line_widths) if all_line_widths else 0,
            'mode_line_width': self._get_mode_with_fallback(all_line_widths) if all_line_widths else 0
        }

    
    
    
    def _determine_line_orientation_by_linewidth(self, x0, y0, x1, y1, line_width):
        """
        使用linewidth属性确定线条方向
        
        Args:
            x0, y0, x1, y1: 线条坐标
            line_width: 线宽属性
            
        Returns:
            'horizontal', 'vertical', 或 None
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        # 如果没有linewidth信息，使用默认值
        if line_width == 0:
            line_width = self.char_analysis['mode_height'] * 0.3 if self.char_analysis['mode_height'] > 0 else self.char_analysis['min_height'] * 0.3
        
        # 使用linewidth作为方向判断的阈值
        tolerance = max(line_width * 2, 3)
        
        if dy <= tolerance and dx > tolerance:
            return 'horizontal'
        elif dx <= tolerance and dy > tolerance:
            return 'vertical'
        else:
            return None
    
    def _analyze_chars(self):
        """
        分析字符特征
        
        计算字符宽度和高度的最小最大值以及众数
        
        Returns:
            None: 结果存储在 self.char_analysis 中
        """
        if not self.chars:
            self.char_analysis = {
                'total_chars': 0,
                'min_width': 0,
                'min_height': 0,
                'max_width': 0,
                'max_height': 0,
                'mode_width': 0,
                'mode_height': 0
            }
            return
        
        # 提取字符宽度和高度（过滤掉无效值）
        widths = [char.get('width', 0) for char in self.chars if char.get('width', 0) > 0]
        heights = [char.get('height', 0) for char in self.chars if char.get('height', 0) > 0]
        
        # 计算统计指标
        self.char_analysis = {
            'total_chars': len(self.chars),
            'min_width': np.min(widths) if widths else 0,
            'min_height': np.min(heights) if heights else 0,
            'max_width': np.max(widths) if widths else 0,
            'max_height': np.max(heights) if heights else 0,
            'mode_width': self._get_mode_with_fallback(widths) if widths else 0,
            'mode_height': self._get_mode_with_fallback(heights) if heights else 0
        }
    
    def _analyze_text_lines(self):
        """分析文本行特征"""
        if not self.text_lines:
            self.text_line_analysis = {
                'total_lines': 0,
                'min_line_height': 0,
                'max_line_height': 0,
                'mode_line_height': 0,
                'min_line_spacing': 0,
                'max_line_spacing': 0,
                'mode_line_spacing': 0
            }
            return
        
        line_heights = []
        line_spacings = []
        
        for text_line in self.text_lines:
            if 'top' in text_line and 'bottom' in text_line:
                line_height = text_line['bottom'] - text_line['top']
                line_heights.append(line_height)
        
        # 计算行间距
        sorted_lines = sorted(self.text_lines, key=lambda x: x.get('top', 0))
        for i in range(len(sorted_lines) - 1):
            current_bottom = sorted_lines[i].get('bottom', 0)
            next_top = sorted_lines[i + 1].get('top', 0)
            spacing = next_top - current_bottom
            if spacing > 0:
                line_spacings.append(spacing)
        
        self.text_line_analysis = {
            'total_lines': len(self.text_lines),
            'min_line_height': np.min(line_heights) if line_heights else 0,
            'max_line_height': np.max(line_heights) if line_heights else 0,
            'mode_line_height': self._get_mode_with_fallback(line_heights) if line_heights else 0,
            'min_line_spacing': np.min(line_spacings) if line_spacings else 0,
            'max_line_spacing': np.max(line_spacings) if line_spacings else 0,
            'mode_line_spacing': self._get_mode_with_fallback(line_spacings) if line_spacings else 0
        }
       
    
    

     
    def _log_page_elements(self):
        """
        输出页面元素的详细调试信息
        
        包括：lines、rects、curves的数量和特征
        """
        self.logger.info("="*70)
        self.logger.info("Page Element Details")
        self.logger.info("="*70)
        
        # 1. Lines信息
        original_lines = self.page.lines if hasattr(self.page, 'lines') else []
        self.logger.info(f"\n1. LINES (Original Lines)")
        self.logger.info(f"   Count: {len(original_lines)}")
        
        if original_lines:
            lengths = []
            for line in original_lines[:10]:  # 只显示前10条
                length = math.sqrt((line['x1'] - line['x0'])**2 + (line['y1'] - line['y0'])**2)
                lengths.append(length)
                self.logger.info(
                    f"   Line: ({line['x0']:.1f},{line['y0']:.1f}) → ({line['x1']:.1f},{line['y1']:.1f}), "
                    f"length={length:.1f}pt, linewidth={line.get('linewidth', 0):.2f}"
                )
            
            if len(original_lines) > 10:
                self.logger.info(f"   ... {len(original_lines) - 10} more lines")
            
            self.logger.info(f"   Length stats: min={min(lengths):.1f}pt, max={max(lengths):.1f}pt, "
                           f"avg={np.mean(lengths):.1f}pt")
        
        # 2. Rects信息
        rects = self.page.rects if hasattr(self.page, 'rects') else []
        self.logger.info(f"\n2. RECTS (Rectangles)")
        self.logger.info(f"   Count: {len(rects)}")
        
        if rects:
            widths = []
            heights = []
            stroked_count = 0
            filled_count = 0
            
            # 获取字符尺寸用于判断
            min_char_width = self.char_analysis.get('min_width', 4.0)
            min_char_height = self.char_analysis.get('min_height', 5.0)
            if min_char_width <= 0:
                min_char_width = 4.0
            if min_char_height <= 0:
                min_char_height = 5.0
            
            for rect in rects[:10]:  # 只显示前10个
                width = rect.get('width', 0)
                height = rect.get('height', 0)
                is_stroked = rect.get('stroke', False)
                is_filled = rect.get('fill', False)
                
                widths.append(width)
                heights.append(height)
                if is_stroked:
                    stroked_count += 1
                if is_filled:
                    filled_count += 1
                
                self.logger.info(
                    f"   Rect: width={width:.2f}pt, height={height:.2f}pt, "
                    f"stroke={is_stroked}, fill={is_filled}"   
                )
            
            if len(rects) > 10:
                self.logger.info(f"   ... {len(rects) - 10} more rectangles")
            
            # 统计所有rects的特征
            all_widths = [r.get('width', 0) for r in rects]
            all_heights = [r.get('height', 0) for r in rects]
            all_stroked = sum(1 for r in rects if r.get('stroke', False))
            all_filled = sum(1 for r in rects if r.get('fill', False))
            
            self.logger.info(f"   Width stats: min={min(all_widths):.2f}pt, max={max(all_widths):.2f}pt, "
                           f"avg={np.mean(all_widths):.2f}pt")
            self.logger.info(f"   Height stats: min={min(all_heights):.2f}pt, max={max(all_heights):.2f}pt, "
                           f"avg={np.mean(all_heights):.2f}pt")
            self.logger.info(f"   Stroked: {all_stroked}/{len(rects)} ({all_stroked/len(rects)*100:.1f}%)")
            self.logger.info(f"   Filled: {all_filled}/{len(rects)} ({all_filled/len(rects)*100:.1f}%)")
        
        # 3. Curves信息
        curves = self.curves
        self.logger.info(f"\n3. CURVES")
        self.logger.info(f"   Count: {len(curves)}")
        
        if curves:
            for curve in curves[:10]:  # 只显示前10条
                points = curve.get('pts', [])
                if len(points) >= 2:
                    x0, y0 = points[0]
                    x1, y1 = points[-1]
                else:
                    x0, y0, x1, y1 = 0, 0, 0, 0
                
                self.logger.info(
                    f"   Curve: ({x0:.1f},{y0:.1f}) → ({x1:.1f},{y1:.1f}), "
                    f"control_points={len(points)}, stroke={curve.get('stroke', False)}"
                )
            
            if len(curves) > 10:
                self.logger.info(f"   ... {len(curves) - 10} more curves")
        
        # 4. 转换后的lines信息
        converted_count = 0
        if self.rects:
            h_lines_from_rects, v_lines_from_rects = self._convert_rects_to_lines(
                char_height_mode=self.char_analysis['min_height'] if self.char_analysis['min_height'] > 0 else 2.0,
                rects=self.rects
            )
            converted_count = len(h_lines_from_rects) + len(v_lines_from_rects)
        self.logger.info(f"\n4. Converted LINES (including rects conversion)")
        self.logger.info(f"   Original lines: {len(original_lines)}, Rects converted lines: {converted_count}")
        

        # 5. 分析后的线条分类
        h_lines = self.line_analysis.get('horizontal_lines', [])
        v_lines = self.line_analysis.get('vertical_lines', [])
        self.logger.info(f"\n5. Line Classification After Analysis")
        self.logger.info(f"   Horizontal lines: {len(h_lines)}")
        self.logger.info(f"   Vertical lines: {len(v_lines)}")        
        self.logger.info("="*70)

        # 6. 分析后的字符信息
        chars_h_min = self.char_analysis.get('min_height', 0)
        chars_w_min = self.char_analysis.get('min_width', 0)
        chars_h_max = self.char_analysis.get('max_height', 0)
        chars_w_max = self.char_analysis.get('max_width', 0)
        chars_h_mode = self.char_analysis.get('mode_height', 0)
        chars_w_mode = self.char_analysis.get('mode_width', 0)
        self.logger.info(f"\n6. Character Information After Analysis")
        self.logger.info(f"   Character height: min={chars_h_min:.2f}pt, max={chars_h_max:.2f}pt, mode={chars_h_mode:.2f}pt")
        
    
    
    
    # === 公开接口：提供访问分析结果的属性 ===
    
    @property
    def char_analysis(self) -> dict:
        """字符分析结果"""
        return self._char_analysis if hasattr(self, '_char_analysis') else {}
    
    @char_analysis.setter
    def char_analysis(self, value):
        self._char_analysis = value
    
    @property
    def line_analysis(self) -> dict:
        """线条分析结果"""
        return self._line_analysis if hasattr(self, '_line_analysis') else {}
    
    @line_analysis.setter
    def line_analysis(self, value):
        self._line_analysis = value
    
    @property
    def text_line_analysis(self) -> dict:
        """文本行分析结果"""
        return self._text_line_analysis if hasattr(self, '_text_line_analysis') else {}
    
    @text_line_analysis.setter
    def text_line_analysis(self, value):
        self._text_line_analysis = value
    
    @property
    def word_analysis(self) -> dict:
        """单词分析结果（已废弃，返回空字典以保持接口兼容性）"""
        return {}
    
    @word_analysis.setter
    def word_analysis(self, value):
        # 不再存储 word_analysis
        pass

