import numpy as np
import pandas as pd
import cv2
import re
from collections import Counter
from scipy.spatial import distance

class TableEvaluator:
    """Domain and source adaptive table quality evaluator"""
    """基于表格来源和模式自适应的表格质量评估器"""

    def __init__(self, domain="unstructured", source="camelot", flavor="lattice"):
        """
        Initialize the table evaluator with domain-specific weights and source configuration
        
        Args:
            domain (str): Domain type - "financial", "scientific", "medical", "unstructured"
            source (str): Table extraction source - "camelot" or "pdfplumber"  
            flavor (str): Camelot extraction mode - "lattice" or "stream" or None
        """
        # Domain-specific weight adjustment factors for different evaluation dimensions
        # Each domain has different priorities: [structural, layout, content, functional]
        self.domain_matrix = {
            "financial": [1.3, 0.9, 1.4, 0.8],    # Emphasize content accuracy and structure
            "scientific": [1.1, 1.0, 1.5, 0.7],   # Emphasize content and layout precision
            "medical": [0.9, 0.8, 1.8, 0.9],      # Emphasize content accuracy
            "unstructured": [1.0, 1.0, 1.0, 1.0]  # Balanced weights
        }
        
        # Base weights for evaluation dimensions
        # when calculating the total score, the weights are adjusted according 
        # to the domain, adjusted = base_weights ⊙ domain_factor, 
        # the adjusted weights are normalized: self.weights = adjusted / adjusted.sum()
        self.base_weights = {
            "structural": 0.35,    # Table structure integrity
            "layout": 0.30,        # Visual layout consistency  
            "content": 0.20,       # Data content quality
            "functional": 0.15     # Practical usability
        }
        
        # Sub-weights for each dimension's components
        self.sub_weights = {
            "structural": [0.5, 0.5],        # [coverage_rate, merge_accuracy]
            "content": [0.5, 0.5],           # [type_consistency, logic_correctness]
            "layout": [0.5, 0.5],            # [row_height_consistency, column_width_stability]
            "functional": [0.7, 0.3]         # [data_extractability, whitespace_control]
        }
        
        self.set_domain(domain)
        self.source = source
        self.flavor = flavor

    def set_domain(self, domain):
        """
        Set evaluation domain and adjust weights accordingly
        
        Args:
            domain (str): Target domain for weight adjustment
        """
        self.domain = domain if domain in self.domain_matrix else "unstructured"
        self.domain_factor = np.array(self.domain_matrix[self.domain])
        # Adjust base weights by domain factors and normalize
        adjusted = np.array(list(self.base_weights.values())) * self.domain_factor
        self.weights = dict(zip(self.base_weights.keys(), adjusted / adjusted.sum()))

    def extract_structural_features(self, table):
        """
        Extract structural features from pdfplumber table object for evaluation
        
        Args:
            table: Table object (Camelot Table or PDFPlumberTableWrapper)
            
        Returns:
            tuple: (DataFrame, features_dict) - Table data and extracted features
        """
        # Extract basic table properties (robust defaults)
        try:
            shape = getattr(table, 'shape', (0, 0)) or (0, 0)
        except Exception:
            shape = (0, 0)
        
        # 安全获取columns和rows，避免pandas Index的布尔判断问题
        columns = getattr(table, 'columns', None)
        if columns is None or (hasattr(columns, 'empty') and columns.empty):
            columns = []
        elif hasattr(columns, 'tolist'):
            columns = columns.tolist()
        
        rows = getattr(table, 'rows', None)
        if rows is None or (hasattr(rows, 'empty') and rows.empty):
            rows = []
        elif hasattr(rows, 'tolist'):
            rows = rows.tolist()
        
        features = {
            'shape': shape,                                      # (rows, cols)
            'accuracy': getattr(table, 'accuracy', 80) or 80,    # 0-100
            'whitespace': getattr(table, 'whitespace', 20) or 20,
            'bbox': getattr(table, 'bbox', None),
            'columns': columns,
            'rows': rows,
            # edges only meaningful for Camelot lattice; wrapper may leave as None
            'edges': getattr(table, 'edges', None),
        }

        
        # Extract cell-level information if available (support Camelot and pdfplumber wrapper)
        if hasattr(table, 'cells') and getattr(table, 'cells'):
            features['cells'] = []
            # cells may be 2D (Camelot) or flat list (wrapper). Normalize iteration
            iterable = getattr(table, 'cells')
            is_2d = isinstance(iterable, (list, tuple)) and len(iterable) > 0 and isinstance(iterable[0], (list, tuple))
            if is_2d:
                cell_iter = (c for row in iterable for c in row)
            else:
                cell_iter = iter(iterable)

            for cell in cell_iter:
                # Wrapper dict format: expects keys bbox/text/edges
                if isinstance(cell, dict):
                    bbox = cell.get('bbox')
                    text = cell.get('text')
                    edges = cell.get('edges')
                    if bbox is None and edges and all(k in edges for k in ('left','right','top','bottom')):
                        # Fallback from edges
                        bbox = (getattr(edges['left'], 'x0', None), getattr(edges['top'], 'y0', None),
                                getattr(edges['right'], 'x1', None), getattr(edges['bottom'], 'y1', None))
                    features['cells'].append({
                        'bbox': bbox,
                        'text': text,
                        'edges': edges
                    })
                else:
                    # Camelot cell object with lb/rt and edge attributes
                    try:
                        bbox = (cell.lb[0], cell.lb[1], cell.rt[0], cell.rt[1])
                    except Exception:
                        # Attempt generic fallbacks
                        x0 = getattr(cell, 'x0', None)
                        y0 = getattr(cell, 'y0', None)
                        x1 = getattr(cell, 'x1', None)
                        y1 = getattr(cell, 'y1', None)
                        bbox = (x0, y0, x1, y1)
                    features['cells'].append({
                        'bbox': bbox,
                        'text': getattr(cell, 'text', None),
                        'edges': {
                            'left': getattr(cell, 'left', None),
                            'right': getattr(cell, 'right', None),
                            'top': getattr(cell, 'top', None),
                            'bottom': getattr(cell, 'bottom', None)
                        }
                    })
        else:
            features['cells'] = None

        # Ensure DataFrame is present
        try:
            # 如果传入的就是DataFrame，直接使用
            if isinstance(table, pd.DataFrame):
                df = table
            else:
                # 否则尝试从对象中获取df属性
                df = getattr(table, 'df', None)
                if df is None:
                    df = pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
        return df, features

    def evaluate(self, table):
        """
        Main evaluation function that computes overall table quality score
        
        Args:
            table: Table object to evaluate
            
        Returns:
            tuple: (total_score, dimension_scores, detected_domain)
                   - total_score: Overall quality score (0-1)
                   - dimension_scores: Scores for each evaluation dimension
                   - detected_domain: Automatically detected domain type
        """
        df,features = self.extract_structural_features(table)
        
        # Validate table structure
        if df.empty or features.get('shape', (0, 0))[0] < 2 or features.get('shape', (0, 0))[1] < 2:
            return 0.0, {"error": "invalid table structure"}, "unknown"
        
        # Auto-detect domain and adjust weights
        domain = "unstructured"
        # domain = self.auto_detect_domain(df, features)
        # self.set_domain(domain)
        # Domain detection and weight adjustment completed
        # Calculate scores for each dimension
        scores = {
            "structural": self._structural_score(df, features),
            "layout": self._layout_score(df,features),
            "content": self._content_score(df, features),
            "functional": self._functional_score(df, features)
        }
        # 调试信息已移除
        # Compute weighted total score
        total_score = sum(scores[dim] * self.weights[dim] for dim in self.weights)
        return round(total_score, 4), scores, domain

    def _structural_score(self, df, features):
        """
        结构评分：覆盖度 × 网格一致性 × 合并合理性 × 基础属性完备度
        - 覆盖度：实际单元与期望网格的匹配程度，结合 df 的非空占比
        - 网格一致性：shape/rows/columns/单元格 bbox 的一致性
        - 合并合理性：仅对 Camelot lattice 使用边框缺失推断合并程度（越合理得分越高）
        - 基础属性完备度：bbox/rows/columns 是否存在
        """
        rows, cols = features.get('shape', (0, 0))
        expected_cells = max(1, rows * cols)

        # 1) 覆盖度（cells 个数 vs 期望；df 非空率）
        cells_list = features.get('cells') or []
        cell_coverage = min(1.0, len(cells_list) / expected_cells) if expected_cells else 0.0
        if df is not None and not df.empty:
            non_null_ratio = 1 - df.isnull().mean().mean()
        else:
            non_null_ratio = 0.7  # 缺省中等
        coverage_score = 0.6 * cell_coverage + 0.4 * non_null_ratio

        # 2) 网格一致性（行高/列宽单调性 + bbox 合法性）
        # rows/columns 为坐标序列：应严格单调
        def _monotonic_ratio(seq):
            try:
                seq = list(seq) if seq is not None else []
                if len(seq) < 2:
                    return 1.0
                diffs = np.diff(seq)
                non_neg = np.mean(diffs > 0)
                return float(non_neg)
            except Exception:
                return 0.7
        columns = features.get('columns') or []
        rows_pos = features.get('rows') or []
        mono_cols = _monotonic_ratio(columns)
        mono_rows = _monotonic_ratio(rows_pos)

        # bbox 合法性（坐标有序且面积>0 的比例）
        valid_bbox_ratio = 1.0
        if cells_list:
            valid = 0
            total = 0
            for c in cells_list:
                bbox = c.get('bbox') if isinstance(c, dict) else None
                if bbox is None and not isinstance(c, dict):
                    try:
                        bbox = (c.lb[0], c.lb[1], c.rt[0], c.rt[1])
                    except Exception:
                        bbox = None
                if bbox is not None:
                    x0, y0, x1, y1 = bbox
                    if x1 is not None and x0 is not None and y1 is not None and y0 is not None:
                        total += 1
                        if x1 > x0 and y1 > y0:
                            valid += 1
            valid_bbox_ratio = (valid / total) if total > 0 else 1.0
        grid_consistency = 0.4 * mono_cols + 0.4 * mono_rows + 0.2 * valid_bbox_ratio

        # 3) 合并合理性（仅 lattice）
        if self.source == "camelot" and (self.flavor or '').lower() == "lattice":
            merge_score = self._detect_merge_cells(features)
        else:
            merge_score = 0.85

        # 4) 基础属性完备度
        base_ok = 0
        base_ok += 1 if features.get('bbox') is not None else 0
        base_ok += 1 if columns else 0
        base_ok += 1 if rows_pos else 0
        base_score = base_ok / 3 if 3 > 0 else 1.0

        # 综合（乘法抑制单项严重短板）
        final = np.prod([
            max(0.05, coverage_score),
            max(0.05, grid_consistency),
            max(0.05, merge_score),
            max(0.05, base_score),
        ]) ** 0.5  # 减弱过度惩罚
        # 调试信息已移除
        return float(max(0.0, min(1.0, final)))

    def _detect_merge_cells(self, features):
        """
        Detect and score merged cells in table structure
        
        Args:
            features: Table structural features
            
        Returns:
            float: Merge detection score (0-1)
        """
        cells = features.get('cells') or []
        if not cells:
            return 0.85
        # 缺边比例：缺水平边或缺垂直边，视为合并迹象
        missing_h = 0
        missing_v = 0
        total = 0
        for c in cells:
            edges = c.get('edges') if isinstance(c, dict) else None
            if edges is None:
                continue
            total += 1
            if edges.get('top') is None or edges.get('bottom') is None:
                missing_h += 1
            if edges.get('left') is None or edges.get('right') is None:
                missing_v += 1
        if total == 0:
            return 0.85
        # 合并“合理性”分数：缺边比例不过高最好（呈 U 型）
        miss_ratio = ((missing_h + missing_v) / (2 * total))
        # 在 0.15~0.35 范围附近得分较高，过低或过高都降低
        center = 0.25
        width = 0.2
        score = np.exp(-((miss_ratio - center) ** 2) / (2 * (width ** 2)))
        return float(max(0.0, min(1.0, score)))
    

    def _layout_score(self, df, features):
        """
        布局评分：行高稳定性 × 列宽稳定性 × 对齐一致性
        - 使用稳健的变异系数（IQR/median）代替纯标准差，降低异常值影响
        - 结合 cells 的 bbox 与行列坐标的偏差评估对齐程度
        """
        # 行高稳定性
        def _robust_cv(values):
            try:
                vals = np.array(values, dtype=float)
                if len(vals) < 2:
                    return 0.0
                q75, q25 = np.percentile(vals, [75 ,25])
                iqr = q75 - q25
                med = np.median(vals)
                if med == 0:
                    return 0.0
                return float((iqr / 1.349) / abs(med))  # IQR to sigma approx
            except Exception:
                return 0.3

        def _extract_row_pos(r):
            # 统一提取行坐标（y 方向）为 float
            try:
                if isinstance(r, (int, float)):
                    return float(r)
                if isinstance(r, (tuple, list)):
                    # 常见为 (x, y) 或 (y0, y1)；优先取第二项
                    if len(r) >= 2 and isinstance(r[1], (int, float)):
                        return float(r[1])
                    if len(r) >= 1 and isinstance(r[0], (int, float)):
                        return float(r[0])
                    return None
                # 对象：尝试 y / top / y0/y1
                for attr in ('y', 'top', 'y0', 'y1'):
                    v = getattr(r, attr, None)
                    if isinstance(v, (int, float)):
                        return float(v)
                y0 = getattr(r, 'y0', None)
                y1 = getattr(r, 'y1', None)
                if isinstance(y0, (int, float)) and isinstance(y1, (int, float)):
                    return float((y0 + y1) / 2)
            except Exception:
                return None
            return None

        def _extract_col_pos(c):
            # 统一提取列坐标（x 方向）为 float
            try:
                if isinstance(c, (int, float)):
                    return float(c)
                if isinstance(c, (tuple, list)):
                    # 常见为 (x, y) 或 (x0, x1)；优先取第一项
                    if len(c) >= 1 and isinstance(c[0], (int, float)):
                        return float(c[0])
                    if len(c) >= 2 and isinstance(c[1], (int, float)):
                        return float(c[1])
                    return None
                for attr in ('x', 'x0', 'left'):
                    v = getattr(c, attr, None)
                    if isinstance(v, (int, float)):
                        return float(v)
                x0 = getattr(c, 'x0', None)
                x1 = getattr(c, 'x1', None)
                if isinstance(x0, (int, float)) and isinstance(x1, (int, float)):
                    return float((x0 + x1) / 2)
            except Exception:
                return None
            return None

        row_positions_raw = features.get('rows') or []
        row_positions = [p for p in (_extract_row_pos(r) for r in row_positions_raw) if isinstance(p, float)]
        row_positions = sorted(row_positions)
        row_heights = [row_positions[i+1] - row_positions[i] for i in range(len(row_positions)-1)]
        cv_h = _robust_cv(row_heights)
        height_score = 1 / (1 + min(2.0, cv_h))

        # 列宽稳定性
        col_positions_raw = features.get('columns') or []
        col_positions = [p for p in (_extract_col_pos(c) for c in col_positions_raw) if isinstance(p, float)]
        col_positions = sorted(col_positions)
        col_widths = [col_positions[i+1] - col_positions[i] for i in range(len(col_positions)-1)]
        cv_w = _robust_cv(col_widths)
        width_score = 1 / (1 + min(2.0, cv_w))

        # 对齐一致性（cell bbox 左右与列线、上下与行线的平均偏差/间距）
        align_score = 1.0
        cells = features.get('cells') or []
        if cells and col_positions and row_positions:
            col_positions_np = np.array(col_positions, dtype=float) if col_positions else np.array([])
            row_positions_np = np.array(row_positions, dtype=float) if row_positions else np.array([])
            devs = []
            for c in cells:
                bbox = c.get('bbox') if isinstance(c, dict) else None
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
                # 与最近列线/行线的距离
                dx = min(abs(col_positions_np - x0).min(), abs(col_positions_np - x1).min()) if len(col_positions_np) else 0
                dy = min(abs(row_positions_np - y0).min(), abs(row_positions_np - y1).min()) if len(row_positions_np) else 0
                # 归一化到平均间距
                avg_w = np.mean(col_widths) if col_widths else (np.max(col_positions_np)-np.min(col_positions_np))/max(1,len(col_positions_np)-1)
                avg_h = np.mean(row_heights) if row_heights else (np.max(row_positions_np)-np.min(row_positions_np))/max(1,len(row_positions_np)-1)
                if avg_w and avg_h:
                    devs.append(0.5*(dx/max(1e-6,avg_w)+dy/max(1e-6,avg_h)))
            if devs:
                dev = np.median(devs)
                align_score = 1 / (1 + min(2.0, dev))

        sub = [height_score, width_score]
        base = float(np.dot(sub, self.sub_weights["layout"]))
        final = (base * 0.7 + align_score * 0.3)
        # 调试信息已移除
        return float(max(0.0, min(1.0, final)))

    def _content_score(self, df, features):
        """
        内容评分：类型一致性 × 逻辑一致性 × 标题/头部合理性
        - 类型一致性：每列主导类型占比（数字/日期/文本），辅以熵正则
        - 逻辑一致性：沿用现有逻辑校验并加入数值非负/单元格数量匹配
        - 头部合理性：首行是否更像表头（文本比例高、去重率高）
        """
        if df is None or df.empty:
            type_score = 0.6
            logic_score = self._check_logic(pd.DataFrame(), features)
            header_score = 0.6
        else:
            # 类型一致性
            type_scores = []
            for col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue
                type_counts = Counter()
                for val in col_data:
                    s = str(val)
                    if re.fullmatch(r"[-+]?\d+(?:[\.,]\d+)?", s.replace(',', '')):
                        type_counts['number'] += 1
                    elif re.match(r'\d{4}[-/年]?(?:\d{1,2})[-/月]?(?:\d{1,2})?', s):
                        type_counts['date'] += 1
                    else:
                        type_counts['text'] += 1
                total = sum(type_counts.values())
                if total > 0:
                    dominant = max(type_counts.values()) / total
                    # 熵（越小越集中，越好）
                    probs = np.array([v/total for v in type_counts.values()], dtype=float)
                    entropy = -np.sum(probs * np.log2(np.clip(probs, 1e-9, 1)))
                    
                    # 修复RuntimeWarning：处理无效的数学运算
                    type_count_len = len(type_counts)
                    if type_count_len <= 1:
                        # 如果只有一种类型或没有类型，熵为0，归一化熵也为0
                        norm_entropy = 0.0
                    else:
                        # 计算归一化熵，避免除以0或log2(0)
                        log2_count = np.log2(type_count_len)
                        if np.isfinite(log2_count) and log2_count > 0:
                            norm_entropy = entropy / log2_count
                        else:
                            norm_entropy = 0.0
                    
                    # 确保norm_entropy是有效数值
                    if not np.isfinite(norm_entropy):
                        norm_entropy = 0.0
                    
                    type_scores.append(0.8*dominant + 0.2*(1-norm_entropy))
            type_score = float(np.mean(type_scores)) if type_scores else 0.7

            # 逻辑一致性（沿用+增强）
            logic_score = self._check_logic(df, features)

            # 头部合理性：第一行文本率与去重率
            try:
                first_row = df.iloc[0].astype(str).tolist()
                text_like = [not re.fullmatch(r"[-+]?\d+(?:[\.,]\d+)?", s.replace(',', '')) for s in first_row]
                text_ratio = np.mean(text_like) if first_row else 0
                unique_ratio = len(set(first_row)) / max(1, len(first_row))
                header_score = 0.5*text_ratio + 0.5*unique_ratio
            except Exception:
                header_score = 0.6

        sub = [type_score, logic_score]
        base = float(np.dot(sub, self.sub_weights["content"]))
        final = (base * 0.8 + header_score * 0.2)
        # 调试信息已移除
        return float(max(0.0, min(1.0, final)))

    def _check_logic(self, df, features):
        """
        Check logical consistency of table data
        
        Args:
            df: Table DataFrame
            features: Extracted structural features
            
        Returns:
            float: Logic score (0-1)
        """
        logic_score = 0.8  # Base score
        
        # Check numeric column constraints
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            # Check for non-negative values in numeric columns
            non_negative = sum(df[col].min() >= 0 for col in numeric_cols) / len(numeric_cols)
            logic_score = max(logic_score, non_negative)
            
        # Check cell structure consistency
        if features.get('shape') and features.get('cells'):
            expected_cells = features['shape'][0] * features['shape'][1]
            actual_cells = len(features['cells'])
            cell_ratio = actual_cells / expected_cells if expected_cells > 0 else 1
            logic_score = max(logic_score, min(1.0, cell_ratio))
            
        return logic_score

    def _functional_score(self, df, features):
        """
        可用性评分：可抽取性 × 空白控制 × 可读性
        - 可抽取性：综合 accuracy 与 df 可解析性（数值解析率、列可用率）
        - 空白控制：平均空白率越低越好，分布越均匀越好
        - 可读性：平均单元文本长度适中、单元填充率合理
        """
        # 可抽取性
        base_extract = self._calculate_extract_score(features)
        parse_bonus = 0.7
        if df is not None and not df.empty:
            try:
                # 列可用率：非空比例 > 60% 的列占比
                usable_cols = 0
                for col in df.columns:
                    non_null = df[col].notnull().mean()
                    if non_null >= 0.6:
                        usable_cols += 1
                col_usable_ratio = usable_cols / max(1, len(df.columns))
                # 数值解析率：数字样式字符串能正确转为数值的比例
                num_like = 0
                num_parsed = 0
                for v in df.astype(str).replace({'': np.nan}).stack():
                    s = str(v)
                    if re.fullmatch(r"[-+]?\d+(?:[\.,]\d+)?", s.replace(',', '')):
                        num_like += 1
                        try:
                            float(s.replace(',', ''))
                            num_parsed += 1
                        except Exception:
                            pass
                parse_ratio = (num_parsed / num_like) if num_like else 0.8
                parse_bonus = 0.5*col_usable_ratio + 0.5*parse_ratio
            except Exception:
                parse_bonus = 0.7
        extract_score = 0.7*base_extract + 0.3*parse_bonus

        # 空白控制
        whitespace_info = self._analyze_whitespace_distribution(features)
        whitespace_level = whitespace_info['avg_whitespace']  # 0~1
        whitespace_even = 1 / (1 + whitespace_info.get('unevenness', 0.0))
        whitespace_score = (1 - min(1.0, whitespace_level)) * 0.7 + whitespace_even * 0.3

        # 可读性（文本密度适中）
        readability = 0.8
        cells = features.get('cells') or []
        if cells:
            lengths = []
            for c in cells:
                t = c.get('text') if isinstance(c, dict) else getattr(c, 'text', '')
                if t is not None:
                    lengths.append(len(str(t)))
            if lengths:
                med = np.median(lengths)
                # 目标中位长度区间 ~ [3, 20]
                if med <= 3:
                    readability = 0.6
                elif med >= 40:
                    readability = 0.6
                elif med <= 10:
                    readability = 0.9
                else:
                    readability = 0.8

        final = float(np.dot([extract_score, whitespace_score], self.sub_weights["functional"]))
        final = 0.8*final + 0.2*readability
        return float(max(0.0, min(1.0, final)))

    def _calculate_extract_score(self, features):
        """
        Calculate data extraction quality score
        
        Args:
            features: Extracted structural features
            
        Returns:
            float: Extraction score (0-1)
        """
        # Combine accuracy and whitespace metrics
        accuracy_score = features.get('accuracy', 80) / 100
        whitespace_score = 1 - min(1, features.get('whitespace', 20) / 50)
        return (accuracy_score * 0.7 + whitespace_score * 0.3)

    def _analyze_whitespace_distribution(self, features):
        """
        Analyze whitespace distribution across table cells
        
        Args:
            features: Extracted structural features
            
        Returns:
            dict: Whitespace analysis results
        """
        if not features.get('whitespace') or not features.get('cells'):
            return {'avg_whitespace': features.get('whitespace', 20)/100, 'unevenness': 0.0}
            
        cell_whitespace = []
        for cell in features['cells']:
            if 'edges' in cell and cell['edges']:
                # Calculate cell area
                cell_area = (cell['bbox'][2]-cell['bbox'][0]) * (cell['bbox'][3]-cell['bbox'][1])
                if cell_area > 0:
                    # Estimate text area based on line count
                    text_height = 10
                    line_count = cell['text'].count('\n') + 1
                    text_area = (cell['bbox'][2]-cell['bbox'][0]) * text_height * line_count
                    whitespace_ratio = max(0, 1 - text_area/cell_area)
                    cell_whitespace.append(whitespace_ratio)
                    
        if not cell_whitespace:
            return {'avg_whitespace': features.get('whitespace', 20)/100, 'unevenness': 0.0}
            
        # Calculate whitespace statistics
        mean_ws = np.mean(cell_whitespace)
        std_ws = np.std(cell_whitespace)
        unevenness = std_ws / mean_ws if mean_ws > 0 else std_ws
        
        return {'avg_whitespace': mean_ws, 'unevenness': unevenness}

    def auto_detect_domain(self, df, features):
        """
        Automatically detect table domain based on content analysis
        
        Args:
            df: Table DataFrame
            features: Extracted structural features
            
        Returns:
            str: Detected domain type
        """
        content = df.astype(str).to_string()
        
        # Financial domain detection
        financial_keywords = ["$", "¥", "total", "amount", "balance"]
        if any(kw in content for kw in financial_keywords):
            return "financial"
            
        # Scientific domain detection
        if any(re.search(r'[αβγΔδ]|p-value|R\^?2', content)):
            return "scientific"
            
        # Medical domain detection
        medical_terms = r'dose|mg/kg|ICD-\d|diagnosis'
        if re.search(medical_terms, content, re.IGNORECASE):
            return "medical"
            
        # Wide tables often indicate scientific data
        if features.get('shape', (0, 0))[1] > 8:
            return "scientific"
            
        return "unstructured"


    def enhance_camelot_features(self, camelot_table):
        """增强 Camelot 表格对象的可评估特征，但保持其原始类型不变。

        - 填充缺失的 bbox/columns/rows（从 Camelot 的属性推导）
        - 从 parsing_report 中提取 accuracy/whitespace 到直达属性
        - 规范 cells：确保存在 text 属性（至少为空字符串）
        """
        table = camelot_table

        # 1) 基本几何信息
        try:
            if getattr(table, 'bbox', None) is None:
                # 优先使用公开属性；退回到受保护属性
                bbox_candidate = getattr(table, 'bbox', None)
                if bbox_candidate is None:
                    bbox_candidate = getattr(table, '_bbox', None)
                if bbox_candidate is not None:
                    try:
                        setattr(table, 'bbox', tuple(bbox_candidate))
                    except Exception:
                        pass
        except Exception:
            pass

        # 2) 行列坐标（Camelot 提供 rows/cols 列表）。为与评估器兼容，补充 columns/rows 属性
        try:
            cols = getattr(table, 'cols', None)
            if (not hasattr(table, 'columns')) or (getattr(table, 'columns', None) in (None, [])):
                if cols is not None:
                    try:
                        setattr(table, 'columns', list(cols))
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            rows = getattr(table, 'rows', None)
            if (not hasattr(table, 'rows')) or (getattr(table, 'rows', None) in (None, [])):
                if rows is not None:
                    try:
                        setattr(table, 'rows', list(rows))
                    except Exception:
                        pass
        except Exception:
            pass

        # 3) 评分报告映射到直达属性
        try:
            report = getattr(table, 'parsing_report', None)
            if isinstance(report, dict):
                if getattr(table, 'accuracy', None) is None and 'accuracy' in report:
                    try:
                        setattr(table, 'accuracy', float(report.get('accuracy', 80)))
                    except Exception:
                        pass
                if getattr(table, 'whitespace', None) is None and 'whitespace' in report:
                    try:
                        setattr(table, 'whitespace', float(report.get('whitespace', 20)))
                    except Exception:
                        pass
        except Exception:
            pass

        # 4) cells 归一化：保证可访问 text 属性。
        try:
            cells = getattr(table, 'cells', None)
            if cells:
                is_2d = isinstance(cells, (list, tuple)) and len(cells) > 0 and isinstance(cells[0], (list, tuple))
                iterable = (c for row in cells for c in row) if is_2d else iter(cells)
                for c in iterable:
                    # Camelot 的单元格通常是对象；确保有 text 属性
                    try:
                        if not hasattr(c, 'text') or getattr(c, 'text') is None:
                            setattr(c, 'text', '')
                    except Exception:
                        # 如果是 dict（不常见于 Camelot），也做兼容
                        if isinstance(c, dict) and 'text' not in c:
                            c['text'] = ''
        except Exception:
            pass

        return table



class PDFPlumberTableWrapper:
    """
    Wrapper class to make PDFPlumber tables compatible with TableEvaluator interface
    Converts PDFPlumber table objects to a format that can be evaluated by TableEvaluator
    """

    def __init__(self, table, page):
        """
        Initialize wrapper with PDFPlumber table and page objects
        
        Args:
            table: PDFPlumber Table object (from page.find_tables())
            page: PDFPlumber Page object
        """
        self.table = table
        self.page = page
        
        # Convert to DataFrame structure (guard None)
        table_data = table.extract() or []
        self.df = pd.DataFrame(table_data)
        # Cache table text (2D list) for reuse when filling cell text
        try:
            self._table_text = self.df.values.tolist()
        except Exception:
            self._table_text = []
        
        self.shape = self.df.shape
        self.bbox = table.bbox if hasattr(table, 'bbox') else None
        self.columns = table.columns if hasattr(table, 'columns') else []
        self.rows = table.rows if hasattr(table, 'rows') else []
        
        # Generate cell structure for evaluation
        self.cells = self.get_pdfplumber_table_cells(table, page)
        
        # Calculate evaluation metrics
        self.accuracy = self.compute_pdfplumber_accuracy(self.cells)
        self.whitespace = self.calculate_pdfplumber_whitespace(self._table_text)
        # Interface compatibility attributes
        

    def get_pdfplumber_table_cells(self, table, page):
        """
        Build cell-level information structure for each table cell
        
        Args:
            table: PDFPlumber Table object
            page: PDFPlumber Page object
            
        Returns:
            list: List of cell dictionaries with bbox, text, and text_bbox information
        """
        table_cells = []
        # Reuse cached table text computed in __init__ to avoid duplicate extract()
        table_text = getattr(self, '_table_text', [])
        
        # Iterate through table rows and cells
        for row_idx, row in enumerate(getattr(table, 'rows', []) or []):
            cells = getattr(row, 'cells', []) or []
            for col_idx, cell_bbox in enumerate(cells):
                # Use cached df-derived text; guard index errors
                cell_text = None
                if cell_bbox is not None:
                    try:
                        cell_text = table_text[row_idx][col_idx]
                    except Exception:
                        cell_text = None
                # Compute text bbox using crop + extract_words for efficiency/clarity
                text_bbox = self.get_cell_text_bbox(page, cell_bbox) if cell_bbox is not None else None
                
                # Build edges approximation (pdfplumber has no explicit cell edges objects)
                edges = None
                if cell_bbox is not None:
                    # 调用calculate_edges_from_bbox
                    edges = self.calculate_edges_from_bbox(cell_bbox)
                    edges = edges if edges is not None else {}

                cell_dict = {
                    "bbox": cell_bbox,
                    "text": cell_text,
                    "text_bbox": text_bbox,
                    "edges": edges,
                }
                table_cells.append(cell_dict)
                
        return table_cells
        

    def get_cell_text_bbox(self, page, cell_bbox):
        """
        Calculate the minimum bounding box of text within a cell
        
        Args:
            page: PDFPlumber Page object
            cell_bbox: Cell bounding box coordinates (x0, y0, x1, y1)
            
        Returns:
            tuple: Text bounding box coordinates or None if no text found
        """
        if not cell_bbox:
            return None
        try:
            cropped = page.crop(cell_bbox)
            words = cropped.extract_words() or []
        except Exception:
            words = []
        if not words:
            return None
        # Calculate minimum bounding box of all words in cell
        def _get(d, *keys):
            for k in keys:
                if k in d:
                    return d[k]
            return 0
        x0 = min(_get(w, "x0", "x_min") for w in words)
        x1 = max(_get(w, "x1", "x_max") for w in words)
        top = min(_get(w, "top", "y0") for w in words)
        bottom = max(_get(w, "bottom", "y1") for w in words)
        return (x0, top, x1, bottom)

    def compute_pdfplumber_accuracy(self, table_cells):
        """
        Compute structural accuracy score similar to Camelot's accuracy metric
        每个文本对象的误差 error 的计算（将文本 bbox 超出单元格左右上下边界的偏移量，
        按文本宽高归一化，得到0~+∞的比例误差；完全贴合时为0）
        
        Args:
            table_cells: List of cell dictionaries
            
        Returns:
            float: Accuracy score (0-100)
        """
        pos_errors = []
        
        for cell in table_cells:
            cell_bbox = cell["bbox"]
            text_bbox = cell.get("text_bbox")
            
            # Handle empty cells or missing text
            if not cell["text"] or not text_bbox or not cell_bbox:
                pos_errors.append(1.0)
                continue
                
            # Calculate positioning errors in x and y directions
            x_error = max(0, cell_bbox[0] - text_bbox[0]) + max(0, text_bbox[2] - cell_bbox[2])
            y_error = max(0, cell_bbox[1] - text_bbox[1]) + max(0, text_bbox[3] - cell_bbox[3])
            
            # Normalize errors by cell dimensions
            cell_w = cell_bbox[2] - cell_bbox[0]
            cell_h = cell_bbox[3] - cell_bbox[1]
            error = (x_error / cell_w + y_error / cell_h) / 2 if cell_w > 0 and cell_h > 0 else 1.0
            
            pos_errors.append(error)
            
        # Convert average error to accuracy percentage
        accuracy = 100 - (np.mean(pos_errors) * 100) if pos_errors else 0
        return accuracy
    

    def calculate_pdfplumber_whitespace(self, table_data):
        """
        Calculate whitespace percentage similar to Camelot's whitespace metric
        
        Args:
            table_data: 2D list of table cell contents
            
        Returns:
            float: Whitespace percentage (0-100)
        """
        total = 0
        blanks = 0
        
        # Count total cells and empty cells
        for row in (table_data or []):
            for cell in (row or []):
                total += 1
                if cell is None or str(cell).strip() == "":
                    blanks += 1
                    
        # Calculate whitespace percentage
        whitespace = (blanks / total) if total else 0.0
        return whitespace * 100

    

    def _derive_columns_from_cells(self, cells):
        """从cells数据推导列位置"""
        if not cells:
            return []
        
        # 按列分组cells，计算每列的平均位置
        col_positions = {}
        
        for i, cell in enumerate(cells):
            if cell.get('bbox'):
                col_idx = i % self.shape[1]  # 假设cells是按行优先排列的
                if col_idx not in col_positions:
                    col_positions[col_idx] = []
                col_positions[col_idx].append(cell['bbox'][0])  # x0位置
        
        # 计算每列的平均x位置
        columns = []
        for col_idx in sorted(col_positions.keys()):
            avg_x = np.mean(col_positions[col_idx])
            columns.append(avg_x)        
        return columns
    
    @staticmethod
    def calculate_edges_from_bbox(cell_bbox):
        """从bbox计算edges信息"""
        if not cell_bbox:
            return None
        
        x0, y0, x1, y1 = cell_bbox
        return {
            'left': {'x0': x0, 'y0': y0, 'x1': x0, 'y1': y1},
            'right': {'x0': x1, 'y0': y0, 'x1': x1, 'y1': y1},
            'top': {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y0},
            'bottom': {'x0': x0, 'y0': y1, 'x1': x1, 'y1': y1}
        }


    
    
    

    

    
   
    
