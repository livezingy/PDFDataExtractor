# core/processing/table_processor.py
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from PIL import Image
import time
from core.utils.logger import AppLogger
from core.models.table_models import TableModels
from core.processing.table_evaluator import TableEvaluator
from core.processing.table_evaluator import PDFPlumberTableWrapper
import pandas as pd
import pdfplumber
import camelot
import numpy as np
from typing import List, Tuple, Dict, Any
from scipy import stats
from collections import defaultdict
import math


class PageFeatureAnalyzer:
    """
    Unified page feature analyzer for extracting and analyzing PDF page characteristics
    Provides comprehensive analysis of lines, text, words, and characters for adaptive parameter setting
    """
    
    def __init__(self, page):
        """
        Initialize page feature analyzer
        
        Args:
            page: pdfplumber Page object
        """
        self.page = page
        self.logger = AppLogger.get_logger()
        
        # Extract and cache all page features
        self.lines = page.lines if hasattr(page, 'lines') else []
        self.chars = page.chars if hasattr(page, 'chars') else []
        self.words = page.extract_words() if hasattr(page, 'extract_words') else []
        self.text_lines = page.extract_text_lines() if hasattr(page, 'extract_text_lines') else []

        # Analyze features
        self._analyze_chars()
        self._analyze_text_lines()
        self._analyze_words()      
        self._analyze_lines()        
        
        
        
    def _analyze_lines(self):
        """Analyze line characteristics including orientation, length, spacing, and distribution"""
        if not self.lines:
            self.line_analysis = {
                'horizontal_lines': [],
                'vertical_lines': [],
                'horizontal_lines_length': [],
                'vertical_lines_length': [],
                'avg_horizontal_length': 0,
                'avg_vertical_length': 0,
                'avg_line_thickness': 0
            }
            return
        
        
        # Calculate minimum length requirement (1.5 * average character height)
        min_length = self.char_analysis['avg_height'] * 1.5

        # Classify valid lines as horizontal or vertical using improved logic
        horizontal_lines = []
        vertical_lines = []       
        horizontal_lengths = []
        vertical_lengths = []
        all_thickness = []
        
        for line in self.lines:
            # Calculate line length
            length = math.sqrt((line['x1'] - line['x0'])**2 + (line['y1'] - line['y0'])**2)
            
            # Keep lines that meet minimum length requirement
            if length >= min_length:
                
                # Calculate line statistics
                x0, y0, x1, y1 = line['x0'], line['y0'], line['x1'], line['y1']
                line_width = line.get('linewidth', 0)
                
                # Determine orientation for statistics calculation
                orientation = self._determine_line_orientation_by_linewidth(x0, y0, x1, y1, line_width)
                
                if orientation == 'horizontal':
                    horizontal_lengths.append(abs(x1 - x0))
                    all_thickness.append(abs(y1 - y0))
                    horizontal_lines.append(line)
                elif orientation == 'vertical':
                    vertical_lengths.append(abs(y1 - y0))
                    all_thickness.append(abs(x1 - x0))
                    vertical_lines.append(line)
        
        
        avg_horizontal_length = np.mean(horizontal_lengths) if horizontal_lengths else 0
        avg_vertical_length = np.mean(vertical_lengths) if vertical_lengths else 0
        avg_line_thickness = np.mean(all_thickness) if all_thickness else 0
   

        # Store basic line classification with statistics
        self.line_analysis = {
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'horizontal_lines_length': horizontal_lengths,
            'vertical_lines_length': vertical_lengths,
            'avg_horizontal_length': avg_horizontal_length,
            'avg_vertical_length': avg_vertical_length,
            'avg_line_thickness': avg_line_thickness
        }

    
    
    def _determine_line_orientation_by_linewidth(self, x0, y0, x1, y1, line_width):
        """
        Determine line orientation using linewidth attribute
        
        Args:
            x0, y0, x1, y1: Line coordinates
            line_width: Line width attribute (use char_height * 0.3 if not available)
            
        Returns:
            'horizontal', 'vertical','ambiguous'
        """
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        
        # Use character height * 0.3 as fallback if no linewidth
        if line_width <= 0:
            line_width = self.char_analysis['avg_height'] * 0.3
        
        # Threshold factor for linewidth comparison
        threshold_factor = 4
        threshold_val = line_width * threshold_factor
        
        # Horizontal line: X direction length >> linewidth, Y direction length ≈ linewidth
        if (height <= threshold_val):# and width/height > threshold_factor):
            return 'horizontal'
        
        # Vertical line: Y direction length >> linewidth, X direction length ≈ linewidth
        elif (width <= threshold_val):# and height/width > threshold_factor):
            return 'vertical'
        else:
            return 'ambiguous'
        
        
    
    def _analyze_chars(self):
        """Analyze character characteristics with outlier removal"""
        if not self.chars:
            self.char_analysis = {
                'avg_width': 10,
                'avg_height': 10
            }
            return
            
        widths = [c['x1'] - c['x0'] for c in self.chars]
        heights = [c['bottom'] - c['top'] for c in self.chars]
        
        # Remove outliers，keep 10%~90% 
        widths_clean = self._remove_outliers(widths)
        heights_clean = self._remove_outliers(heights)
        
        self.char_analysis = {
            'avg_width': np.mean(widths_clean) if widths_clean else np.mean(widths),
            'avg_height': np.mean(heights_clean) if heights_clean else np.mean(heights)
        }
    
    def _analyze_text_lines(self):
        """Analyze text line characteristics"""
        if not self.text_lines:
            self.text_line_analysis = {
                'avg_line_height': 10,
                'avg_line_gap': 5,
                'total_lines': 0
            }
            return
            
        # Calculate line heights
        line_heights = [line['bottom'] - line['top'] for line in self.text_lines]
        
        # Calculate line gaps
        tops = sorted([line['top'] for line in self.text_lines])
        line_gaps = [tops[i] - tops[i-1] for i in range(1, len(tops))]

        # Remove outliers，keep 10%~90% 
        line_heights_clean = self._remove_outliers(line_heights)
        line_gaps_clean = self._remove_outliers(line_gaps)
        
        self.text_line_analysis = {
            'avg_line_height': np.mean(line_heights_clean) if line_heights_clean else np.mean(line_heights),
            'avg_line_gap': np.mean(line_gaps_clean) if line_gaps_clean else np.mean(line_gaps),
            'total_lines': len(self.text_lines)
        }
    
    def _analyze_words(self):
        """Analyze word spacing characteristics using text_line-based approach"""
        if not self.text_lines:
            self.word_analysis = {
                'avg_word_gap': 10,
                'avg_word_width': 20,
                'avg_word_height': 15
            }
            return
        
        all_word_gaps = []
        all_word_widths = []
        all_word_heights = []
        
        # Process each text line individually
        for text_line in self.text_lines:
            # Get words within this text line's bounding box
            # text_line has 'top', 'bottom' properties and 'chars' list
            if 'chars' in text_line and text_line['chars']:
                # Use first and last character coordinates to define bbox
                first_char = text_line['chars'][0]
                last_char = text_line['chars'][-1]
                bbox = (first_char['x0'], text_line['top'], last_char['x1'], text_line['bottom'])
            else:
                # Fallback: use top/bottom with estimated width
                estimated_width = self.char_analysis['avg_width'] * 50  # rough estimate
                bbox = (0, text_line['top'], estimated_width, text_line['bottom'])
            
            line_words = self.page.within_bbox(bbox).extract_words()
            
            if len(line_words) < 2:
                continue  # Skip lines with less than 2 words
            
            # Sort words by x-coordinate within this line
            line_words_sorted = sorted(line_words, key=lambda w: w['x0'])
            
            # Calculate word gaps within this line
            for i in range(1, len(line_words_sorted)):
                gap = line_words_sorted[i]['x0'] - line_words_sorted[i-1]['x1']
                if gap > 0:  # Only positive gaps
                    all_word_gaps.append(gap)
            
            # Collect word dimensions
            for word in line_words:
                all_word_widths.append(word['x1'] - word['x0'])
                all_word_heights.append(word['bottom'] - word['top'])
        
        # Remove outliers and calculate statistics
        word_gaps_clean = self._remove_outliers(all_word_gaps)
        word_widths_clean = self._remove_outliers(all_word_widths)
        word_heights_clean = self._remove_outliers(all_word_heights)
        
        self.word_analysis = {
            'avg_word_gap': np.median(word_gaps_clean) if word_gaps_clean else 0,
            'avg_word_width': np.mean(word_widths_clean) if word_widths_clean else np.mean(all_word_widths),
            'avg_word_height': np.mean(word_heights_clean) if word_heights_clean else np.mean(all_word_heights)
        }   
    
    
    def _calculate_min_line_spacing(self, lines, coord_key):
        """Calculate minimum spacing between lines"""
        if len(lines) < 2:
            return float('inf')
            
        coords = sorted([line[coord_key] for line in lines])
        spacings = [coords[i] - coords[i-1] for i in range(1, len(coords))]
        return min(spacings) if spacings else float('inf')

    
    def _calculate_min_endpoint_distance(self):
        """Calculate minimum distance between line endpoints"""
        if len(self.lines) < 2:
            return float('inf')
            
        endpoints = []
        for line in self.lines:
            endpoints.extend([(line['x0'], line['y0']), (line['x1'], line['y1'])])
        
        min_distances = []
        for i, ep1 in enumerate(endpoints):
            min_dist = float('inf')
            for j, ep2 in enumerate(endpoints):
                if i != j:
                    dist = math.sqrt((ep1[0] - ep2[0])**2 + (ep1[1] - ep2[1])**2)
                    min_dist = min(min_dist, dist)
            if min_dist < float('inf'):
                min_distances.append(min_dist)
        
        return min(min_distances) if min_distances else float('inf')    
    
    
    
    def predict_table_type(self):
        """
        基于思路B的表格类型预测：使用比例聚集度判断
        
        思路B核心：
        1. 使用25%-75%分位数确定主要区域
        2. 计算线条集中度 = 主要区域内线条数 / 总线条数
        3. 计算区域集中度 = 主要区域面积 / 页面面积
        4. 使用两个比例的加权和进行最终判断
        """
        # 1. 基础检查：线条数量是否充足
        horizontal_lines = self.line_analysis['horizontal_lines']
        vertical_lines = self.line_analysis['vertical_lines']
        
        if len(horizontal_lines) < 3 or len(vertical_lines) < 3:
            return 'unbordered'
        
        # 2. 收集所有线条坐标
        all_lines = horizontal_lines + vertical_lines
        all_x = []
        all_y = []
        for line in all_lines:
            all_x.extend([line['x0'], line['x1']])
            all_y.extend([line['y0'], line['y1']])
        
        # 3. 计算25%-75%分位数确定主要区域
        x_q25, x_q75 = np.percentile(all_x, [25, 75])
        y_q25, y_q75 = np.percentile(all_y, [25, 75])
        
        # 4. 计算主要区域面积和页面面积
        main_region_area = (x_q75 - x_q25) * (y_q75 - y_q25)
        page_area = self.page.width * self.page.height
        
        # 5. 统计主要区域内的线条数量
        main_region_lines = 0
        for line in all_lines:
            # 检查线条是否与主要区域重叠
            line_x_min = min(line['x0'], line['x1'])
            line_x_max = max(line['x0'], line['x1'])
            line_y_min = min(line['y0'], line['y1'])
            line_y_max = max(line['y0'], line['y1'])
            
            # 判断线条是否与主要区域有重叠
            if (line_x_max >= x_q25 and line_x_min <= x_q75 and
                line_y_max >= y_q25 and line_y_min <= y_q75):
                main_region_lines += 1
        
        # 6. 计算两个关键比例
        line_concentration = main_region_lines / len(all_lines)  # 线条集中度
        area_ratio = main_region_area / page_area  # 区域集中度
        
        # 7. 使用加权和进行最终判断
        # 线条集中度权重0.7，区域集中度权重0.3
        # 线条集中度高且区域集中度低表示有框表格
        final_score = line_concentration * 0.7 + (1.0 - area_ratio) * 0.3
        self.logger.info(f"[TableProcessor] bordered table type score: {final_score}")
        
        # 8. 判断阈值
        is_bordered = final_score > 0.6
        
        return 'bordered' if is_bordered else 'unbordered'

    
    #lattice模式需要添加block_size参数
    def _calculate_adaptive_short_line_thresholds(self, h_lengths, v_lengths):
        """
        自适应短线阈值计算
        
        结合多种方法，选择最合适的阈值
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

    def get_camelot_lattice_params(self, image_shape=None):
        """改进的Camelot lattice参数计算"""
        params = {
            'flavor': 'lattice',
            'line_scale': 40,
            'line_tol': 2,
            'joint_tol': 2
        }
        
        h_lines = self.line_analysis['horizontal_lines']
        v_lines = self.line_analysis['vertical_lines']
        
        # 直接使用已存储的线条长度
        h_lengths = self.line_analysis['horizontal_lines_length']
        v_lengths = self.line_analysis['vertical_lines_length']
        
        # 使用动态阈值计算短线比例
        h_short_threshold, v_short_threshold = self._calculate_adaptive_short_line_thresholds(h_lengths, v_lengths)
        
        short_h_count = len([l for l in h_lengths if l < h_short_threshold])
        short_v_count = len([l for l in v_lengths if l < v_short_threshold])
        short_h_ratio = short_h_count / len(h_lines) if h_lines else 0
        short_v_ratio = short_v_count / len(v_lines) if v_lines else 0
        
        # 记录阈值信息用于调试
        self.logger.info(f"Dynamic thresholds - H: {h_short_threshold:.2f}, V: {v_short_threshold:.2f}")
        self.logger.info(f"Short line ratios - H: {short_h_ratio:.2f}, V: {short_v_ratio:.2f}")
        
        # 1. 计算 line_scale
        if image_shape and h_lengths and v_lengths:
            if short_h_ratio > 0.3 or short_v_ratio > 0.3:
                # 有较多短线，使用固定的较大值
                params['line_scale'] = 40
            else:
                # 使用自适应计算
                avg_line_length = (np.mean(h_lengths) + np.mean(v_lengths)) / 2
                pdf_to_image_ratio = min(image_shape[0] / self.page.height, 
                                        image_shape[1] / self.page.width)
                line_scale = int(image_shape[0] / (avg_line_length * pdf_to_image_ratio * 0.5))
                params['line_scale'] = max(15, min(line_scale, 50))
        
        # 2. 计算 line_tol
        min_h_spacing = self._calculate_min_line_spacing(h_lines, 'y0')
        min_v_spacing = self._calculate_min_line_spacing(v_lines, 'x0')
        
        if min_h_spacing < 0.1 and min_v_spacing < 0.1:
            # 精确绘制的PDF，使用字符宽度作为基准
            line_tol = self.char_analysis['avg_width'] * 0.3
        else:
            line_tol = min(min_h_spacing * 0.15 if min_h_spacing < float('inf') else 3,
                          min_v_spacing * 0.15 if min_v_spacing < float('inf') else 3,
                          self.char_analysis['avg_width'] * 0.5)
        
        params['line_tol'] = max(0.5, min(line_tol, 3))
        
        # 3. 计算 joint_tol
        min_endpoint_dist = self._calculate_min_endpoint_distance()
        
        if min_endpoint_dist < float('inf'):
            if short_h_ratio > 0.3 or short_v_ratio > 0.3:
                # 有短线，使用更保守的joint_tol
                params['joint_tol'] = max(2, min(min_endpoint_dist * 0.5, 5))
            else:
                params['joint_tol'] = max(1, min(min_endpoint_dist * 1.2, 10))
        
        return params
    
    def get_camelot_stream_params(self):
        """Calculate Camelot stream parameters based on page features"""
        params = {
            'flavor': 'stream',
            'edge_tol': 50,
            'row_tol': 2,
            'column_tol': 0
        }
        
        # edge_tol based on text line gaps
        if self.text_line_analysis['avg_line_gap'] > 0:
            edge_tol = self.text_line_analysis['avg_line_gap'] * 3 + self.text_line_analysis['avg_line_height'] * 2
            params['edge_tol'] = max(20, min(edge_tol, 800))
        
        # row_tol based on line height
        if self.text_line_analysis['avg_line_height'] > 0:
            row_tol = self.text_line_analysis['avg_line_height'] * 1.5
            params['row_tol'] = max(1, min(row_tol, 10))
        
        # column_tol based on word gaps
        if self.word_analysis['avg_word_gap'] > 0:
            column_tol = min(self.word_analysis['avg_word_gap'] * 0.5, 
                           self.char_analysis['avg_width'] * 1.5)
            params['column_tol'] = max(0, min(column_tol, 5))
        
        return params

    
    def get_pdfplumber_params(self, table_type='bordered'):
        """Calculate pdfplumber parameters based on page features and table type"""
        if table_type == 'bordered':
            params = {
                'vertical_strategy': 'lines',
                'horizontal_strategy': 'text',
                'snap_tolerance': 2,
                'join_tolerance': 2,
                'edge_min_length': 3,
                'intersection_tolerance': 3,
                'min_words_vertical': 1,
                'min_words_horizontal': 1,
                'text_x_tolerance': 3,
                'text_y_tolerance': 5
            }
        else:
            params = {
                'vertical_strategy': 'text',
                'horizontal_strategy': 'text',
                'snap_tolerance': 2,
                'join_tolerance': 2,
                'edge_min_length': 3,
                'intersection_tolerance': 3,
                'min_words_vertical': 1,
                'min_words_horizontal': 1,
                'text_x_tolerance': 3,
                'text_y_tolerance': 5
            }
        
        # Calculate adaptive parameters
        # "snap_tolerance", "snap_x_tolerance", "snap_y_tolerance"：Parallel lines within 
        # snap_tolerance points will be "snapped" to the same horizontal or vertical position.
        if self.char_analysis['avg_width'] > 0:
            snap_tol = min(self.char_analysis['avg_width'], self.char_analysis['avg_height']) * 0.5
            params['snap_tolerance'] = max(1, min(snap_tol, 5))
        
        # Calculate endpoint distances
        min_endpoint_dist = self._calculate_min_endpoint_distance()             

        # "join_tolerance", "join_x_tolerance", "join_y_tolerance": Line segments on the same infinite
        #  line, and whose ends are within join_tolerance of one another, will be "joined" into a single line segment.
        if min_endpoint_dist < float('inf'):
            join_tol = min_endpoint_dist * 1.2
            params['join_tolerance'] = max(1, min(join_tol, 10))
        
        # edge_min_length： Edges shorter than edge_min_length will be discarded 
        # before attempting to reconstruct the table.
        if self.char_analysis['avg_width'] > 0:
            edge_min_len = max(self.char_analysis['avg_width'], self.char_analysis['avg_height']) * 1.2
            params['edge_min_length'] = max(3, min(edge_min_len, 20))
        
        # When using "vertical_strategy": "text", at least min_words_vertical words must 
        # share the same alignment.
        if self.text_line_analysis['total_lines'] > 0:
            min_words_vertical = max(1, int(self.text_line_analysis['total_lines'] * 0.2))
            params['min_words_vertical'] = min_words_vertical


        # When using "horizontal_strategy": "text", at least min_words_horizontal words must 
        # share the same alignment.

        # These text_-prefixed settings also apply to the table-identification algorithm when
        # the text strategy is used. I.e., when that algorithm searches for words, 
        # it will expect the individual letters in each word to be no more than 
        # text_x_tolerance/text_y_tolerance points apart.
        # 具体来说，算法期望每个单词中相邻字母在水平方向（x/y方向）上相隔的距离不超过“text_x/y_tolerance”点
        # （这里的“点”是一种距离单位，可能用于衡量字符间距等）
        if self.word_analysis['avg_word_gap'] > 0:
            text_x_tol = min(self.word_analysis['avg_word_gap'] * 0.8, 
                           self.char_analysis['avg_width'] * 1.5)
            params['text_x_tolerance'] = max(1, min(text_x_tol, 10))
        
        if self.char_analysis['avg_height'] > 0:
            text_y_tol = self.char_analysis['avg_height'] * 0.5
            params['text_y_tolerance'] = max(1, min(text_y_tol, 10))
        
        return params

    
    
    def _remove_outliers(self, data):
        """
        Remove outliers 
        
        Args:
            data: List of numerical values
            
        Returns:
            List of values with outliers removed
        """
        if len(data) < 4:  # Need at least 4 values for meaningful IQR
            return data
            
        data_sorted = sorted(data)
        q10 = np.percentile(data_sorted, 10)
        q90 = np.percentile(data_sorted, 90)        
        
        return [x for x in data if q10 <= x <= q90]


class TableProcessor:
    """Optimized processor with dependency injection"""
    
    def __init__(self, params: Optional[Dict] = None):
        self.logger = AppLogger.get_logger()
        self.params = params or {}
        # 推荐：从params获取已初始化的models
        self.models = self.params.get('models')


    def process_pdf_page(self, pdf_path, page):
        """
        Process PDF page with adaptive parameter setting based on page features
        
        Args:
            pdf_path: Path to PDF file
            page: pdfplumber Page object
            
        Returns:
            List of extracted tables with scores and metadata
        """
        try:
            # 参数有效性检查
            if not page:
                self.logger.error("Page object is None")
                return []
            
            if not pdf_path or not isinstance(pdf_path, (str, Path)):
                self.logger.error(f"Invalid pdf_path: {pdf_path}")
                return []

            # Initialize page feature analyzer with error handling
            try:
                feature_analyzer = PageFeatureAnalyzer(page)
            except Exception as e:
                self.logger.error(f"Failed to initialize PageFeatureAnalyzer: {e}")
                return []
            
            # Get method and flavor from params with validation
            method = self.params.get("table_method", "mixed").lower()
            if method not in ['camelot', 'pdfplumber', 'mixed']:
                self.logger.error(f"Invalid table_method: {method}. Must be one of: camelot, pdfplumber, mixed")
                return []
            
            flavor = self.params.get("table_flavor", None)  # None means auto-detect
            score_threshold = self.params.get("table_score_threshold", 0.5)
            
            # 验证score_threshold范围
            if not isinstance(score_threshold, (int, float)) or score_threshold < 0 or score_threshold > 1:
                self.logger.warning(f"Invalid score_threshold: {score_threshold}. Using default 0.5")
                score_threshold = 0.5
            
            page_num = getattr(page, "page_number", 1)
            
            # Auto-detect table type and flavor if not specified
            if flavor is None:
                try:
                    table_type = feature_analyzer.predict_table_type()
                    
                    if method == "pdfplumber":
                        flavor = "lines" if table_type == "bordered" else "text"
                    elif method == "camelot":
                        flavor = "lattice" if table_type == "bordered" else "stream"
                    else:  # mixed method
                        flavor = "auto"  # Will be determined dynamically
                except Exception as e:
                    self.logger.error(f"Failed to predict table type: {e}")
                    return []
            
            self.logger.info(f"[TableProcessor] Method: {method}, Flavor: {flavor}, Table type: {feature_analyzer.predict_table_type()} on page {page_num}")
            
            # Process based on method and flavor
            try:
                if method == "pdfplumber":
                    return self._process_pdfplumber(page, feature_analyzer, flavor, score_threshold)
                elif method == "camelot":
                    return self._process_camelot(pdf_path, page, feature_analyzer, flavor, score_threshold)
                elif method == "mixed":
                    return self._process_mixed(pdf_path, page, feature_analyzer, score_threshold)
                else:
                    self.logger.error(f"Unknown table extraction method: {method}")
                    return []
            except Exception as e:
                self.logger.error(f"Error during table processing: {e}")
                return []
                
        except Exception as e:
            self.logger.error(f"Unexpected error in process_pdf_page: {e}")
            return []
    
    def _process_pdfplumber(self, page, feature_analyzer, flavor, score_threshold):
        """Process page using pdfplumber with adaptive parameters"""
        table_type = feature_analyzer.predict_table_type()
        
        if flavor == "lines" or (flavor is None and table_type == "bordered"):
            results = self.extract_pdfplumber_lines(page, feature_analyzer)
        elif flavor == "text" or (flavor is None and table_type == "unbordered"):
            results = self.extract_pdfplumber_text(page, feature_analyzer)
        else:
            self.logger.error(f"Unknown pdfplumber flavor: {flavor}")
            return []
        
        # Filter by score threshold
        results = [r for r in results if r["score"] >= score_threshold]
        return results

    def _process_camelot(self, pdf_path, page, feature_analyzer, flavor, score_threshold):
        """Process page using camelot with adaptive parameters"""
        page_num = getattr(page, "page_number", 1)
        table_type = feature_analyzer.predict_table_type()
        
        if flavor == "lattice" or (flavor is None and table_type == "bordered"):
            results = self.extract_camelot_lattice(pdf_path, page_num, page, feature_analyzer)
        elif flavor == "stream" or (flavor is None and table_type == "unbordered"):
            results = self.extract_camelot_stream(pdf_path, page_num, page, feature_analyzer)
        else:
            self.logger.error(f"Unknown camelot flavor: {flavor}")
            return []
        
        # Filter by score threshold
        results = [r for r in results if r["score"] >= score_threshold]
        return results

    def _process_mixed(self, pdf_path, page, feature_analyzer, score_threshold):
        """Process page using mixed method with adaptive parameters"""
        # First pass: pdfplumber detection
        pdfplumber_lines = self.extract_pdfplumber_lines(page, feature_analyzer)
        pdfplumber_text = self.extract_pdfplumber_text(page, feature_analyzer)
        all_pdfplumber = pdfplumber_lines + pdfplumber_text
        
        # Get high-score regions for camelot refinement
        high_score_bboxes = [r["bbox"] for r in all_pdfplumber if r["score"] > 0.7 and r["bbox"] is not None]
        page_num = getattr(page, "page_number", 1)
        
        # Second pass: camelot refinement
        camelot_results = []
        if high_score_bboxes:
            table_type = feature_analyzer.predict_table_type()
            if table_type == "bordered":
                camelot_results = self.extract_camelot_lattice(pdf_path, page_num, page, feature_analyzer, table_areas=high_score_bboxes)
            else:
                camelot_results = self.extract_camelot_stream(pdf_path, page_num, page, feature_analyzer, table_areas=high_score_bboxes)
        
        # Merge and deduplicate results
        all_results = all_pdfplumber + camelot_results
        unique_tables = {}
        for item in all_results:
            bbox_key = tuple(np.round(item['bbox'], 2)) if item['bbox'] is not None else None
            if bbox_key not in unique_tables or item['score'] > unique_tables[bbox_key]['score']:
                unique_tables[bbox_key] = item
        
        final_tables = [v for v in unique_tables.values() if v['score'] >= score_threshold]
        self.logger.debug(f"[TableProcessor] Final tables after deduplication and thresholding: {len(final_tables)}")
        return final_tables

    def extract_pdfplumber_lines(self, page, feature_analyzer=None) -> list:
        """Extract bordered tables using pdfplumber lines mode with adaptive parameters"""
        evaluator = TableEvaluator()
        evaluator.source = "pdfplumber"
        evaluator.flavor = "lines"

        # Get adaptive parameters from feature analyzer
        if feature_analyzer is None:
            feature_analyzer = PageFeatureAnalyzer(page)
        
        params = feature_analyzer.get_pdfplumber_params('bordered')
        tables = page.find_tables(params)
        
        self.logger.info(f"[TableProcessor] PDFPlumber (lines) detected {len(tables)} tables on page {getattr(page, 'page_number', '?')}")
        self.logger.debug(f"[TableProcessor] Using parameters: {params}")
        
        results = []
        for idx, p_table in enumerate(tables):
            wrapper = PDFPlumberTableWrapper(p_table, page) 
            p_score, p_details, p_domain = evaluator.evaluate(wrapper)
            self.logger.info(f"[TableProcessor] PDFPlumber lines table {idx+1}: score={p_score:.3f}, domain={p_domain}, bbox={getattr(p_table, 'bbox', None)}")
            results.append({
                'table': wrapper,
                'bbox': p_table.bbox,
                'score': p_score,
                'details': p_details,
                'domain': p_domain,
                'source': 'pdfplumber_lines'
            })
        return results

    def extract_pdfplumber_text(self, page, feature_analyzer=None) -> list:
        """Extract unbordered tables using pdfplumber text mode with adaptive parameters"""
        evaluator = TableEvaluator()
        evaluator.source = "pdfplumber"
        evaluator.flavor = "text"

        # Get adaptive parameters from feature analyzer
        if feature_analyzer is None:
            feature_analyzer = PageFeatureAnalyzer(page)
        
        params = feature_analyzer.get_pdfplumber_params('unbordered')
        tables = page.find_tables(params)
        
        self.logger.info(f"[TableProcessor] PDFPlumber (text) detected {len(tables)} tables on page {getattr(page, 'page_number', '?')}")
        self.logger.debug(f"[TableProcessor] Using parameters: {params}")
        
        results = []
        for idx, p_table in enumerate(tables):
            wrapper = PDFPlumberTableWrapper(p_table, page)
            p_score, p_details, p_domain = evaluator.evaluate(wrapper)
            self.logger.info(f"[TableProcessor] PDFPlumber text table {idx+1}: score={p_score:.3f}, domain={p_domain}, bbox={getattr(p_table, 'bbox', None)}")
            results.append({
                'table': wrapper,
                'bbox': p_table.bbox,
                'score': p_score,
                'details': p_details,
                'domain': p_domain,
                'source': 'pdfplumber_text'
            })
        return results

    def extract_camelot_lattice(self, pdf_path, page_num, page, feature_analyzer=None, table_areas=None) -> list:
        """Extract tables using Camelot lattice mode with adaptive parameters"""
        evaluator = TableEvaluator()
        evaluator.source = "camelot"
        evaluator.flavor = "lattice"

        # Get adaptive parameters from feature analyzer
        if feature_analyzer is None:
            feature_analyzer = PageFeatureAnalyzer(page)
        
        # Get image shape for coordinate conversion (simplified approach)
        # In practice, you might want to get actual image dimensions
        image_shape = (int(page.height * 2), int(page.width * 2))  # Assume 2x scaling
        
        params = feature_analyzer.get_camelot_lattice_params(image_shape)
        params['pages'] = str(page_num)
        self.logger.info(f"[TableProcessor] Using camelot lattice parameters: {params}")
        
        if table_areas:
            params['table_areas'] = [",".join(map(str, area)) for area in table_areas]
        
        self.logger.debug(f"[TableProcessor] Using lattice parameters: {params}")
        
        try:
            camelot_tables = camelot.read_pdf(pdf_path, **params)
        except Exception as e:
            self.logger.error(f"Camelot lattice extraction failed: {str(e)}")
            return []
        
        self.logger.info(f"[TableProcessor] Camelot (lattice) detected {len(camelot_tables)} tables on page {page_num}")
        results = []
        for idx, ct in enumerate(camelot_tables):
            en_ct = evaluator.enhance_camelot_features(ct)
            c_score, c_details, c_domain = evaluator.evaluate(en_ct)
            self.logger.info(f"[TableProcessor] Camelot lattice table {idx+1}: score={c_score:.3f}, domain={c_domain}, bbox={getattr(en_ct, 'bbox', None)}")
            results.append({
                'table': en_ct,
                'bbox': getattr(en_ct, 'bbox', None),
                'score': c_score,
                'details': c_details,
                'domain': c_domain,
                'source': 'camelot_lattice'
            })
        return results


    def extract_camelot_stream(self, pdf_path, page_num, page, feature_analyzer=None, table_areas=None) -> list:
        """Extract tables using Camelot stream mode with adaptive parameters"""
        evaluator = TableEvaluator()
        evaluator.source = "camelot"
        evaluator.flavor = "stream"

        # Get adaptive parameters from feature analyzer
        if feature_analyzer is None:
            feature_analyzer = PageFeatureAnalyzer(page)
        
        params = feature_analyzer.get_camelot_stream_params()
        params['pages'] = str(page_num)
        
        if table_areas:
            params['table_areas'] = [",".join(map(str, area)) for area in table_areas]
        
        self.logger.debug(f"[TableProcessor] Using stream parameters: {params}")
        
        try:
            camelot_tables = camelot.read_pdf(pdf_path, **params)
        except Exception as e:
            self.logger.error(f"Camelot stream extraction failed: {str(e)}")
            return []
        
        self.logger.info(f"[TableProcessor] Camelot (stream) detected {len(camelot_tables)} tables on page {page_num}")
        results = []
        for idx, ct in enumerate(camelot_tables):
            en_ct = evaluator.enhance_camelot_features(ct)
            c_score, c_details, c_domain = evaluator.evaluate(en_ct)
            self.logger.info(f"[TableProcessor] Camelot stream table {idx+1}: score={c_score:.3f}, domain={c_domain}, bbox={getattr(en_ct, 'bbox', None)}")
            results.append({
                'table': en_ct,
                'bbox': getattr(en_ct, 'bbox', None),
                'score': c_score,
                'details': c_details,
                'domain': c_domain,
                'source': 'camelot_stream'
            })
        return results


    
    

   



    




