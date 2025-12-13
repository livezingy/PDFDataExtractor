# Camelot参数计算原理

本文档详细说明PDF Table Extractor中Camelot参数自动计算的原理和实现方法。

## 概述

Camelot是Python中用于从PDF中提取表格的库，支持两种主要模式：
- **Lattice模式**：适用于有明确边框的表格
- **Stream模式**：适用于无边框或边框不完整的表格

系统通过分析PDF页面的特征（线条、文本、字符等），自动计算最适合的Camelot参数。

## 参数计算实现

### 1. Lattice模式参数计算

**方法位置**：`core/processing/table_processor.py` - `PageFeatureAnalyzer.get_camelot_lattice_params()`

#### 核心参数

| 参数 | 计算依据 | 公式 |
|------|----------|------|
| `line_scale` | 线条密度和页面尺寸 | `min(40, max(15, int(min(image_shape) * 0.01)))` |
| `shift_text` | 字符和线条位置关系 | 基于字符bbox与线条的包含关系 |
| `copy_text` | 文本分布特征 | 基于文本行数量和分布 |

#### 计算逻辑

```python
def get_camelot_lattice_params(self, image_shape=None):
    """计算Camelot lattice模式参数"""
    params = {
        'flavor': 'lattice',
        'line_scale': 40,
        'shift_text': ['l'],
        'copy_text': ['l']
    }
    
    # 1. line_scale计算
    if image_shape:
        line_scale = min(40, max(15, int(min(image_shape) * 0.01)))
        params['line_scale'] = line_scale
    
    # 2. shift_text计算
    if self.char_analysis['total_chars'] > 0:
        # 分析字符与线条的关系
        char_line_overlap = self._analyze_char_line_overlap()
        if char_line_overlap > 0.3:  # 30%以上字符与线条重叠
            params['shift_text'] = ['l', 'r', 't', 'b']
    
    # 3. copy_text计算
    if self.text_line_analysis['total_lines'] > 0:
        # 基于文本行分布决定是否复制文本
        text_density = self.text_line_analysis['total_lines'] / self.page_area
        if text_density > 0.01:  # 文本密度较高
            params['copy_text'] = ['l', 'r', 't', 'b']
    
    return params
```

### 2. Stream模式参数计算

**方法位置**：`core/processing/table_processor.py` - `PageFeatureAnalyzer.get_camelot_stream_params()`

#### 核心参数

| 参数 | 计算依据 | 公式 |
|------|----------|------|
| `edge_tol` | 文本行间距 | `行间距 × 3 + 行高 × 2` |
| `row_tol` | 行高 | `行高 × 1.5` |
| `column_tol` | 词间距 | `词间距 × 0.5` |

#### 计算逻辑

```python
def get_camelot_stream_params(self):
    """计算Camelot stream模式参数"""
    params = {
        'flavor': 'stream',
        'edge_tol': 50,
        'row_tol': 2,
        'column_tol': 0
    }
    
    # 1. edge_tol计算
    if self.text_line_analysis['avg_line_gap'] > 0:
        edge_tol = (self.text_line_analysis['avg_line_gap'] * 3 + 
                   self.text_line_analysis['avg_line_height'] * 2)
        params['edge_tol'] = max(20, min(edge_tol, 800))
    
    # 2. row_tol计算
    if self.text_line_analysis['avg_line_height'] > 0:
        row_tol = self.text_line_analysis['avg_line_height'] * 1.5
        params['row_tol'] = max(1, min(row_tol, 10))
    
    # 3. column_tol计算
    if self.word_analysis['avg_word_gap'] > 0:
        column_tol = min(self.word_analysis['avg_word_gap'] * 0.5, 
                        self.char_analysis['avg_width'] * 1.5)
        params['column_tol'] = max(0, min(column_tol, 5))
    
    return params
```

## 页面特征分析

### 1. 线条分析

系统会分析PDF页面中的线条特征：

```python
def _analyze_lines(self, page):
    """分析页面线条特征"""
    lines = page.lines
    line_analysis = {
        'total_lines': len(lines),
        'horizontal_lines': len([l for l in lines if abs(l['y1'] - l['y2']) < 1]),
        'vertical_lines': len([l for l in lines if abs(l['x1'] - l['x2']) < 1]),
        'avg_line_length': np.mean([l['length'] for l in lines]) if lines else 0,
        'line_density': len(lines) / self.page_area
    }
    return line_analysis
```

### 2. 文本行分析

分析文本行的分布和特征：

```python
def _analyze_text_lines(self, page):
    """分析文本行特征"""
    text_lines = []
    for line in page.lines:
        if line['text']:
            text_lines.append({
                'text': line['text'],
                'bbox': [line['x0'], line['y0'], line['x1'], line['y1']],
                'height': line['y1'] - line['y0'],
                'width': line['x1'] - line['x0']
            })
    
    if text_lines:
        heights = [tl['height'] for tl in text_lines]
        gaps = [text_lines[i+1]['bbox'][1] - text_lines[i]['bbox'][3] 
                for i in range(len(text_lines)-1)]
        
        return {
            'total_lines': len(text_lines),
            'avg_line_height': np.mean(heights),
            'avg_line_gap': np.mean(gaps) if gaps else 0,
            'line_height_std': np.std(heights)
        }
    return {'total_lines': 0, 'avg_line_height': 0, 'avg_line_gap': 0, 'line_height_std': 0}
```

### 3. 字符分析

分析字符的分布特征：

```python
def _analyze_characters(self, page):
    """分析字符特征"""
    chars = page.chars
    if not chars:
        return {'total_chars': 0, 'avg_width': 0, 'avg_height': 0}
    
    widths = [c['width'] for c in chars]
    heights = [c['height'] for c in chars]
    
    return {
        'total_chars': len(chars),
        'avg_width': np.mean(widths),
        'avg_height': np.mean(heights),
        'width_std': np.std(widths),
        'height_std': np.std(heights)
    }
```

## 参数优化策略

### 1. 边界值控制

所有计算出的参数都有合理的边界值：

```python
# line_scale: 15-40之间
line_scale = min(40, max(15, calculated_value))

# edge_tol: 20-800之间
edge_tol = max(20, min(calculated_value, 800))

# row_tol: 1-10之间
row_tol = max(1, min(calculated_value, 10))

# column_tol: 0-5之间
column_tol = max(0, min(calculated_value, 5))
```

### 2. 异常值处理

使用分位数方法移除异常值：

```python
def _remove_outliers(self, data):
    """移除异常值，保留10%-90%分位数的数据"""
    if len(data) < 3:
        return data
    
    q10 = np.percentile(data, 10)
    q90 = np.percentile(data, 90)
    return [x for x in data if q10 <= x <= q90]
```

### 3. 自适应调整

根据页面特征动态调整参数：

```python
# 根据线条密度调整line_scale
if line_density > 0.1:  # 高密度线条
    line_scale = min(line_scale, 30)
elif line_density < 0.01:  # 低密度线条
    line_scale = max(line_scale, 25)

# 根据文本密度调整copy_text
if text_density > 0.02:  # 高文本密度
    params['copy_text'] = ['l', 'r', 't', 'b']
```

## 使用示例

```python
# 创建页面特征分析器
analyzer = PageFeatureAnalyzer(page)

# 获取lattice模式参数
lattice_params = analyzer.get_camelot_lattice_params(image_shape=(1200, 800))

# 获取stream模式参数
stream_params = analyzer.get_camelot_stream_params()

# 使用参数提取表格
tables = camelot.read_pdf(pdf_path, **lattice_params)
```

## 注意事项

1. **图像尺寸影响**：`line_scale`参数与图像尺寸相关，需要传入正确的图像尺寸
2. **文本密度**：高文本密度页面可能需要调整`copy_text`参数
3. **线条质量**：低质量线条可能需要降低`line_scale`值
4. **参数验证**：建议在使用前验证计算出的参数是否合理

## 后续优化方向

1. **多页面分析**：考虑整个文档的表格特征分布
2. **用户反馈**：根据提取结果质量动态调整参数
3. **A/B测试**：对比不同参数组合的提取效果
