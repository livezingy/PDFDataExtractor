# core/processing/table_type_classifier.py
"""
表格类型分类器

职责：
- 基于PageFeatureAnalyzer的分析结果判断表格类型
- 实现多层级判断逻辑（快速过滤、对齐检查、统计特征）
- 返回'bordered'或'unbordered'

依赖：
- PageFeatureAnalyzer (通过构造函数注入)
"""

from typing import Literal
import numpy as np
from core.utils.logger import AppLogger


class TableTypeClassifier:
    """
    表格类型分类器
    
    基于页面特征分析结果判断表格类型（有框/无框）
    """
    
    def __init__(self, feature_analyzer, page):
        """
        初始化分类器
        
        Args:
            feature_analyzer: PageFeatureAnalyzer实例（已完成分析）
            page: pdfplumber.Page对象（用于获取页面尺寸等信息）
        """
        self.analyzer = feature_analyzer
        self.page = page
        self.logger = AppLogger.get_logger()
    
    
    def predict_table_type(self) -> Literal['bordered', 'unbordered']:
        """
        预测表格类型：多层级判断
        
        改进点：
        1. 快速过滤层（线条数量）
        2. 对齐度检查（线条极多时）
        3. 使用MAD代替分位数（更鲁棒）
        4. 增加方向平衡性权重
        
        Returns:
            'bordered': 有框表格
            'unbordered': 无框表格
        """
        horizontal_lines = self.analyzer.line_analysis['horizontal_lines']
        vertical_lines = self.analyzer.line_analysis['vertical_lines']
        h_count = len(horizontal_lines)
        v_count = len(vertical_lines)
        
        # ========== 层级1：快速过滤（线条数量） ==========
        
        # 情况1：线条明显不足 → 直接判定unbordered
        if h_count < 3 or v_count < 3:
            self.logger.info(
                f"[Quick Judgment] Insufficient lines (H:{h_count}, V:{v_count}) → unbordered"
            )
            return 'unbordered'
        
        # 情况2：线条极多且高度对齐 → 直接判定bordered
        # 使用动态阈值：基于页面尺寸计算
        dynamic_threshold = self._calculate_dynamic_threshold()
        if h_count > dynamic_threshold and v_count > dynamic_threshold:
            # 简单对齐度检查
            h_aligned = self._quick_alignment_check(horizontal_lines, 'horizontal')
            v_aligned = self._quick_alignment_check(vertical_lines, 'vertical')
            
            if h_aligned > 0.6 and v_aligned > 0.6:
                self.logger.info(
                    f"[Quick Judgment] Many lines and aligned (H:{h_aligned:.2f}, V:{v_aligned:.2f}, "
                    f"line count H:{h_count}, V:{v_count}, threshold:{dynamic_threshold}) → bordered"
                )
                return 'bordered'
        
        # ========== 层级2：标准判断（使用MAD改进） ==========
        # 当线条数量在中等范围（3-动态阈值条）或对齐度不够时
        
        all_lines = horizontal_lines + vertical_lines
        all_x = []
        all_y = []
        for line in all_lines:
            all_x.extend([line['x0'], line['x1']])
            all_y.extend([line['y0'], line['y1']])
        
        # 使用MAD（中位数绝对偏差）代替分位数 - 更鲁棒
        x_median = np.median(all_x)
        y_median = np.median(all_y)
        
        x_mad = np.median([abs(x - x_median) for x in all_x])
        y_mad = np.median([abs(y - y_median) for y in all_y])
        
        # 计算集中区域（median ± 1.5*MAD）
        x_lower = x_median - 1.5 * x_mad
        x_upper = x_median + 1.5 * x_mad
        y_lower = y_median - 1.5 * y_mad
        y_upper = y_median + 1.5 * y_mad
        
        # 计算集中区域面积
        main_region_area = (x_upper - x_lower) * (y_upper - y_lower)
        page_area = self.page.width * self.page.height
        
        # 统计集中区域内的线条数量
        main_region_lines = 0
        for line in all_lines:
            line_x_min = min(line['x0'], line['x1'])
            line_x_max = max(line['x0'], line['x1'])
            line_y_min = min(line['y0'], line['y1'])
            line_y_max = max(line['y0'], line['y1'])
            
            # 判断线条是否与集中区域重叠
            if (line_x_max >= x_lower and line_x_min <= x_upper and
                line_y_max >= y_lower and line_y_min <= y_upper):
                main_region_lines += 1
        
        # 计算关键指标
        line_concentration = main_region_lines / len(all_lines)  # 线条集中度
        area_ratio = main_region_area / page_area  # 区域集中度
        
        # 计算方向平衡性（新增）
        direction_balance = min(h_count, v_count) / max(h_count, v_count)
        
        # 调整后的评分公式（三个维度）
        final_score = (
            line_concentration * 0.6 +      # 线条集中度（降低权重）
            (1.0 - area_ratio) * 0.2 +      # 区域集中度（降低权重）
            direction_balance * 0.2          # 方向平衡性（新增）
        )
        
        self.logger.info(
            f"[Standard Judgment] concentration={line_concentration:.2f}, "
            f"area_ratio={area_ratio:.2f}, balance={direction_balance:.2f}, "
            f"score={final_score:.2f}"
        )
        
        is_bordered = final_score > 0.6
        
        return 'bordered' if is_bordered else 'unbordered'
    
    
    def _quick_alignment_check(self, lines, direction):
        """
        快速对齐度检查（简化版，不使用DBSCAN）
        
        时间复杂度：O(n log n) - 排序主导
        
        原理：将相近坐标（tolerance内）分组，检查最大组占比
        
        Args:
            lines: 线条列表
            direction: 'horizontal' 或 'vertical'
            
        Returns:
            float: 对齐度评分 (0-1)，1表示完全对齐
        """
        if len(lines) < 3:
            return 0.0
        
        # 提取坐标
        coord_key = 'y0' if direction == 'horizontal' else 'x0'
        coords = sorted([line[coord_key] for line in lines])
        
        # 简单分组：相邻坐标差小于tolerance则归为一组
        tolerance = 3  # 固定tolerance，快速判断
        
        groups = []
        current_group = [coords[0]]
        
        # 单次遍历分组
        for i in range(1, len(coords)):
            if coords[i] - current_group[-1] <= tolerance:
                current_group.append(coords[i])
            else:
                groups.append(current_group)
                current_group = [coords[i]]
        groups.append(current_group)
        
        # 计算最大组占比
        max_group_size = max(len(g) for g in groups)
        alignment_ratio = max_group_size / len(coords)
        
        return alignment_ratio
    
    def _calculate_dynamic_threshold(self):
        """
        计算动态阈值：基于页面尺寸自适应调整
        
        原理：
        - 页面越大，可能容纳的表格越大，阈值相应提高
        - 基于A4页面（595x842pt）作为基准，阈值设为10
        - 其他页面按面积比例缩放
        
        Returns:
            int: 动态阈值（最小10，最大30）
        """
        # 基准：A4页面（595x842pt）的阈值为10
        a4_width = 595.0
        a4_height = 842.0
        a4_area = a4_width * a4_height
        base_threshold = 10
        
        # 计算当前页面面积
        page_area = self.page.width * self.page.height
        
        # 按面积比例计算阈值
        area_ratio = page_area / a4_area
        
        # 动态阈值：基准阈值 * 面积比例
        # 使用平方根缩放，避免大页面阈值过高
        dynamic_threshold = int(base_threshold * np.sqrt(area_ratio))
        
        # 限制在合理范围内 [10, 30]
        dynamic_threshold = max(10, min(dynamic_threshold, 30))
        
        self.logger.debug(
            f"Dynamic threshold calculation: page size={self.page.width:.1f}x{self.page.height:.1f}pt, "
            f"area ratio={area_ratio:.2f}, threshold={dynamic_threshold}"
        )
        
        return dynamic_threshold

