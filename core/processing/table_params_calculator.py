# core/processing/table_params_calculator.py
"""
表格参数计算器

职责：
- 基于PageFeatureAnalyzer的分析结果计算自适应参数
- 支持pdfplumber和Camelot（lattice + stream）
- 参数验证和边界检查

依赖：
- PageFeatureAnalyzer (通过构造函数注入)
"""

from typing import Dict, Optional, Tuple
import numpy as np
from core.utils.logger import AppLogger


class TableParamsCalculator:
    """
    表格参数计算器
    
    基于页面特征分析结果计算pdfplumber和Camelot的自适应参数
    """
    
    def __init__(self, feature_analyzer):
        """
        初始化参数计算器
        
        Args:
            feature_analyzer: PageFeatureAnalyzer实例（已完成分析）
        """
        self.analyzer = feature_analyzer
        self.page = feature_analyzer.page
        self.logger = AppLogger.get_logger()
    
    
    def get_pdfplumber_params(self, table_type: str = 'bordered') -> Dict:
        """
        计算pdfplumber的自适应参数
        
        改进版：动态选择strategy，自适应系数计算
        
        Args:
            table_type: 'bordered' 或 'unbordered'
            
        Returns:
            dict: pdfplumber参数字典
        """
        # 获取线条信息
        h_lines = self.analyzer.line_analysis.get('horizontal_lines', [])
        v_lines = self.analyzer.line_analysis.get('vertical_lines', [])
        h_count = len(h_lines)
        v_count = len(v_lines)
        
        # 动态选择strategy
        if table_type == 'bordered':
            vertical_strategy = 'lines' if v_count >= 5 else 'text'
            horizontal_strategy = 'lines' if h_count >= 10 else 'text'
            
            self.logger.debug(
                f"Bordered table strategy: V={vertical_strategy}(count={v_count}), "
                f"H={horizontal_strategy}(count={h_count})"
            )
        else:
            vertical_strategy = 'text'
            horizontal_strategy = 'lines' if h_count >= 3 else 'text'
            
            self.logger.debug(
                f"Unbordered table strategy: V={vertical_strategy}, "
                f"H={horizontal_strategy}(count={h_count})"
            )
        
        # 基础参数
        params = {
            'vertical_strategy': vertical_strategy,
            'horizontal_strategy': horizontal_strategy,
            # Parallel lines within snap_tolerance points will be merged to the same 
            # horizontal or vertical position.
            'snap_tolerance': 2,
            # Line segments on the same infinite line, and whose ends are within 
            # join_tolerance of one another, will be joined into a single line segment.
            'join_tolerance': 2,
            # The minimum length of a line segment that is considered to be part of a table edge.
            'edge_min_length': 3,
            # When combining edges into cells, orthogonal edges must be within 
            # intersection_tolerance points to be considered intersecting.
            'intersection_tolerance': 3,
            #When using "vertical_strategy": "text", at least min_words_vertical words must share the same alignment.
            'min_words_vertical': 1,
            #When using "horizontal_strategy": "text", at least min_words_horizontal words must share the same alignment.
            'min_words_horizontal': 1,
            #These text_-prefixed settings also apply to the table-identification algorithm when the text strategy is used. 
            # I.e., when that algorithm searches for words, it will expect the individual letters in each word to be
            #  no more than text_x_tolerance/text_y_tolerance points apart.
            'text_x_tolerance': 3,
            'text_y_tolerance': 5
        }
        
        # 自适应snap_tolerance[0.5, 5]
        if self.analyzer.char_analysis['min_width'] > 0 and self.analyzer.char_analysis['min_height'] > 0:
            min_char_size = min(self.analyzer.char_analysis['min_width'], 
                               self.analyzer.char_analysis['min_height'])
            params['snap_tolerance'] = min_char_size * 0.3
            params['snap_tolerance'] = max(0.5, min(params['snap_tolerance'], 5))
            
            self.logger.debug(
                f"snap_tolerance={params['snap_tolerance']:.2f} "
                f"(min_char_size={min_char_size:.2f})"
            )
        
        # 自适应join_tolerance[1, 10]
        if self.analyzer.char_analysis['min_width'] > 0 and self.analyzer.char_analysis['min_height'] > 0:
            min_char_size = min(self.analyzer.char_analysis['min_width'], 
                               self.analyzer.char_analysis['min_height'])
            params['join_tolerance'] = min_char_size * 0.3
            params['join_tolerance'] = max(1, min(params['join_tolerance'], 10))
            
            self.logger.debug(
                f"join_tolerance={params['join_tolerance']:.2f} "
                f"(min_char_size={min_char_size:.2f})"
            )
        
        # 自适应edge_min_length（区分表格类型）[1, 30]
        if self.analyzer.char_analysis['mode_width'] > 0 and self.analyzer.char_analysis['mode_height'] > 0:
            mode_char_size = max(self.analyzer.char_analysis['mode_width'], 
                                self.analyzer.char_analysis['mode_height'])
            params['edge_min_length'] = mode_char_size
            params['edge_min_length'] = max(1, min(params['edge_min_length'], 30))
            
            self.logger.debug(
                f"edge_min_length={params['edge_min_length']:.2f} "
                f"(mode_char_size={mode_char_size:.2f})"
            )
        elif self.analyzer.char_analysis['min_width'] > 0 or self.analyzer.char_analysis['min_height'] > 0:
            min_char_size = max(self.analyzer.char_analysis['min_width'], 
                               self.analyzer.char_analysis['min_height'])
            params['edge_min_length'] = min_char_size
            params['edge_min_length'] = max(1, min(params['edge_min_length'], 30))
            
            self.logger.debug(
                f"edge_min_length={params['edge_min_length']:.2f} "
                f"(min_char_size={min_char_size:.2f})"
            )
        
        # 自适应intersection_tolerance[1, 10]
        if self.analyzer.char_analysis['mode_width'] > 0 or self.analyzer.char_analysis['mode_height'] > 0:
            max_mode_size = max(self.analyzer.char_analysis['mode_width'], 
                               self.analyzer.char_analysis['mode_height'])
            params['intersection_tolerance'] = max_mode_size * 0.5
            params['intersection_tolerance'] = max(1, min(params['intersection_tolerance'], 10))
            
            self.logger.debug(
                f"intersection_tolerance={params['intersection_tolerance']:.2f} "
                f"(max_mode_size={max_mode_size:.2f})"
            )
        elif self.analyzer.char_analysis['min_width'] > 0 or self.analyzer.char_analysis['min_height'] > 0:
            max_min_size = max(self.analyzer.char_analysis['min_width'], 
                              self.analyzer.char_analysis['min_height'])
            params['intersection_tolerance'] = max_min_size * 0.5
            params['intersection_tolerance'] = max(1, min(params['intersection_tolerance'], 10))
            
            self.logger.debug(
                f"intersection_tolerance={params['intersection_tolerance']:.2f} "
                f"(max_min_size={max_min_size:.2f})"
            )
        
        # 自适应min_words_vertical[3, 10]
        if self.analyzer.text_line_analysis['total_lines'] > 0:
            min_words = int(self.analyzer.text_line_analysis['total_lines'] * 0.2)
            params['min_words_vertical'] = max(3, min(min_words, 10))
            
            self.logger.debug(
                f"min_words_vertical={params['min_words_vertical']} "
                f"(total_lines={self.analyzer.text_line_analysis['total_lines']})"
            )
        
        # 自适应text_x_tolerance[1, 10]
        if self.analyzer.char_analysis['mode_width'] > 0:
            params['text_x_tolerance'] = self.analyzer.char_analysis['mode_width'] * 1.5
            params['text_x_tolerance'] = max(1, min(params['text_x_tolerance'], 10))
            
            self.logger.debug(
                f"text_x_tolerance={params['text_x_tolerance']:.2f} "
                f"(mode_width={self.analyzer.char_analysis['mode_width']:.2f})"
            )
        elif self.analyzer.char_analysis['min_width'] > 0:
            params['text_x_tolerance'] = self.analyzer.char_analysis['min_width'] * 1.5
            params['text_x_tolerance'] = max(1, min(params['text_x_tolerance'], 10))
            
            self.logger.debug(
                f"text_x_tolerance={params['text_x_tolerance']:.2f} "
                f"(min_width={self.analyzer.char_analysis['min_width']:.2f})"
            )
        
        # 自适应text_y_tolerance[1, 8]
        if self.analyzer.text_line_analysis.get('min_line_height', 0) > 0:
            params['text_y_tolerance'] = self.analyzer.text_line_analysis['min_line_height'] * 0.2
            params['text_y_tolerance'] = max(1, min(params['text_y_tolerance'], 8))
            
            self.logger.debug(
                f"text_y_tolerance={params['text_y_tolerance']:.2f} "
                f"(min_line_height={self.analyzer.text_line_analysis['min_line_height']:.2f})"
            )
        
        # 参数验证
        params = self._validate_params(params)
        
        self.logger.debug(f"Final pdfplumber params: {params}")
        
        return params
    
    
    def get_camelot_lattice_params(self, image_shape: Optional[Tuple] = None) -> Dict:
        """
        计算Camelot lattice模式参数
        
        Args:
            image_shape: 图像尺寸 (height, width)，可选
            
        Returns:
            dict: Camelot lattice参数字典
        """
        params = {
            'flavor': 'lattice',
            'line_scale': 40,
            'line_tol': 2,
            'joint_tol': 2
        }
        
        h_lines = self.analyzer.line_analysis['horizontal_lines']
        v_lines = self.analyzer.line_analysis['vertical_lines']
        h_lengths = self.analyzer.line_analysis['horizontal_lines_length']
        v_lengths = self.analyzer.line_analysis['vertical_lines_length']
        
        # 计算短线阈值
        h_short_threshold, v_short_threshold = self._calculate_adaptive_short_line_thresholds(
            h_lengths, v_lengths
        )
        
        short_h_count = len([l for l in h_lengths if l < h_short_threshold])
        short_v_count = len([l for l in v_lengths if l < v_short_threshold])
        short_h_ratio = short_h_count / len(h_lines) if h_lines else 0
        short_v_ratio = short_v_count / len(v_lines) if v_lines else 0
        
        self.logger.info(f"Dynamic thresholds - H: {h_short_threshold:.2f}, V: {v_short_threshold:.2f}")
        self.logger.info(f"Short line ratios - H: {short_h_ratio:.2f}, V: {short_v_ratio:.2f}")
        
        # 计算line_scale（基于线条宽度众数）
        if image_shape:
            line_widths = self.analyzer.line_analysis.get('line_widths', [])
            if line_widths and len(line_widths) > 0:
                # 计算线条宽度众数
                mode_line_width = self.analyzer._get_mode_with_fallback(line_widths)
                if mode_line_width > 0:
                    # 计算PDF到图像的缩放比例
                    pdf_to_image_ratio = min(image_shape[0] / self.page.height, 
                                            image_shape[1] / self.page.width)
                    
                    # 将PDF坐标的线条宽度转换为图像坐标
                    mode_line_width_image = mode_line_width * pdf_to_image_ratio
                    
                    # 期望kernel长度 = 线条宽度 * 3（根据要求）
                    desired_kernel_length = mode_line_width_image * 3
                    
                    # 计算垂直和水平方向的line_scale
                    if desired_kernel_length > 0:
                        line_scale_v = image_shape[0] / desired_kernel_length
                        line_scale_h = image_shape[1] / desired_kernel_length
                        # 取较小值（保守策略）
                        line_scale = min(line_scale_v, line_scale_h)
                        params['line_scale'] = max(15, min(int(line_scale), 50))
                        
                        self.logger.debug(
                            f"line_scale={params['line_scale']} "
                            f"(mode_line_width={mode_line_width:.2f}pt, "
                            f"mode_line_width_image={mode_line_width_image:.2f}px, "
                            f"desired_kernel_length={desired_kernel_length:.2f}px)"
                        )
                    else:
                        params['line_scale'] = 40  # 默认值
                else:
                    params['line_scale'] = 40  # 默认值
            else:
                params['line_scale'] = 40  # 默认值
        
        # 计算line_tol = joint_tol = min(min_width, min_height) × 0.3
        if self.analyzer.char_analysis['min_width'] > 0 and self.analyzer.char_analysis['min_height'] > 0:
            min_char_size = min(self.analyzer.char_analysis['min_width'], 
                               self.analyzer.char_analysis['min_height'])
            line_tol = min_char_size * 0.3
            params['line_tol'] = max(0.5, min(line_tol, 3))
            params['joint_tol'] = params['line_tol']  # joint_tol = line_tol
            
            self.logger.debug(
                f"line_tol=joint_tol={params['line_tol']:.2f} "
                f"(min_char_size={min_char_size:.2f})"
            )
        else:
            params['line_tol'] = 2  # 默认值
            params['joint_tol'] = 2  # 默认值
        
        return params
    
    
    def get_camelot_stream_params(self) -> Dict:
        """
        计算Camelot stream模式参数
        
        Returns:
            dict: Camelot stream参数字典
        """
        params = {
            'flavor': 'stream',
            'edge_tol': 50,
            'row_tol': 2,
            'column_tol': 0
        }
        
        # edge_tol: 最小值设置为行间距最小值+行高最大值；最大值设置为1/3页面高度
        min_line_spacing = self.analyzer.text_line_analysis.get('min_line_spacing', 0)
        max_line_height = self.analyzer.text_line_analysis.get('max_line_height', 0)
        
        if min_line_spacing > 0 or max_line_height > 0:
            # 计算最小值：行间距最小值 + 行高最大值
            edge_tol_min = min_line_spacing + max_line_height
            
            # 计算最大值：1/3页面高度
            edge_tol_max = self.page.height / 3
            
            # 使用当前计算公式（保持当前逻辑）
            if self.analyzer.text_line_analysis.get('mode_line_spacing', 0) > 0:
                mode_line_spacing = self.analyzer.text_line_analysis['mode_line_spacing']
                mode_line_height = self.analyzer.text_line_analysis.get('mode_line_height', 0)
                if mode_line_height > 0:
                    edge_tol = mode_line_spacing * 3 + mode_line_height * 2
                else:
                    edge_tol = mode_line_spacing * 3 + max_line_height * 2
            else:
                # 回退到最小值
                edge_tol = edge_tol_min
            
            # 限制在最小值到最大值之间
            params['edge_tol'] = max(edge_tol_min, min(edge_tol, edge_tol_max))
            
            self.logger.debug(
                f"edge_tol={params['edge_tol']:.2f} "
                f"(min={edge_tol_min:.2f}, max={edge_tol_max:.2f}, "
                f"calculated={edge_tol:.2f})"
            )
        else:
            # 默认值
            params['edge_tol'] = 50
        
        # row_tol: 最小字符高度
        if self.analyzer.char_analysis['min_height'] > 0:
            params['row_tol'] = self.analyzer.char_analysis['min_height']
            params['row_tol'] = max(1, min(params['row_tol'], 10))
            
            self.logger.debug(
                f"row_tol={params['row_tol']:.2f} "
                f"(min_char_height={self.analyzer.char_analysis['min_height']:.2f})"
            )
        else:
            params['row_tol'] = 2  # 默认值
        
        # column_tol: 最小字符宽度
        if self.analyzer.char_analysis['min_width'] > 0:
            params['column_tol'] = self.analyzer.char_analysis['min_width']
            params['column_tol'] = max(0, min(params['column_tol'], 5))
            
            self.logger.debug(
                f"column_tol={params['column_tol']:.2f} "
                f"(min_char_width={self.analyzer.char_analysis['min_width']:.2f})"
            )
        else:
            params['column_tol'] = 0  # 默认值
        
        return params
    
    
    def _calculate_adaptive_short_line_thresholds(self, h_lengths, v_lengths) -> Tuple[float, float]:
        """
        自适应短线阈值计算
        
        结合多种方法，选择最合适的阈值
        
        Args:
            h_lengths: 水平线长度列表
            v_lengths: 垂直线长度列表
            
        Returns:
            tuple: (水平线阈值, 垂直线阈值)
        """
        if not h_lengths or not v_lengths:
            return 200, 150
        
        # 方法1：分位数方法
        h_q25 = np.percentile(h_lengths, 25)
        v_q25 = np.percentile(v_lengths, 25)
        
        # 方法2：中位数比例方法
        h_median = np.median(h_lengths)
        v_median = np.median(v_lengths)
        h_median_thresh = h_median * 0.6
        v_median_thresh = v_median * 0.6
        
        # 方法3：标准差方法
        h_mean = np.mean(h_lengths)
        v_mean = np.mean(v_lengths)
        h_std = np.std(h_lengths)
        v_std = np.std(v_lengths)
        h_std_thresh = max(h_mean - h_std, h_mean * 0.4)
        v_std_thresh = max(v_mean - v_std, v_mean * 0.4)
        
        # 综合选择：取三个方法的中位数
        h_candidates = [h_q25, h_median_thresh, h_std_thresh]
        v_candidates = [v_q25, v_median_thresh, v_std_thresh]
        
        h_threshold = np.median(h_candidates)
        v_threshold = np.median(v_candidates)
        
        # 设置合理范围
        h_threshold = max(30, min(h_threshold, h_median * 0.8))
        v_threshold = max(30, min(v_threshold, v_median * 0.8))
        
        return h_threshold, v_threshold
    
    
    def _validate_params(self, params: dict) -> dict:
        """
        参数边界检查和修正
        
        确保所有参数在合理范围内，避免极端值导致的错误
        
        Args:
            params: 待验证的参数字典
            
        Returns:
            dict: 验证并修正后的参数
        """
        bounds = {
            'snap_tolerance': (0.5, 10),
            'join_tolerance': (1, 10),
            'edge_min_length': (1, 30),
            'intersection_tolerance': (1, 10),
            'min_words_vertical': (3, 10),
            'min_words_horizontal': (1, 5),
            'text_x_tolerance': (1, 10),
            'text_y_tolerance': (1, 8)
        }
        
        validated = params.copy()
        
        for key, (min_val, max_val) in bounds.items():
            if key in validated:
                original_val = validated[key]
                validated[key] = max(min_val, min(validated[key], max_val))
                
                if validated[key] != original_val:
                    self.logger.debug(
                        f"Parameter {key} adjusted: {original_val:.2f} → {validated[key]:.2f} "
                        f"(bounds: [{min_val}, {max_val}])"
                    )
        
        return validated

