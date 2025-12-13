# 表格提取器使用文档

## 概述

本文档说明如何单独使用Camelot和PDFPlumber提取器模块。这些模块已经模块化，可以独立使用或移植到其他项目中。

## 目录结构

```
core/extractors/
├── __init__.py          # 模块初始化，自动注册提取器
├── base.py              # BaseExtractor基类
├── factory.py           # ExtractorFactory工厂类
├── camelot_extractor.py # Camelot提取器
└── pdfplumber_extractor.py # PDFPlumber提取器
```

## 快速开始

### 1. 基本使用

```python
from core.extractors.factory import ExtractorFactory
from core.processing.page_feature_analyzer import PageFeatureAnalyzer
import pdfplumber

# 打开PDF
with pdfplumber.open('example.pdf') as pdf:
    page = pdf.pages[0]
    
    # 创建特征分析器
    feature_analyzer = PageFeatureAnalyzer(page, enable_logging=False)
    
    # 使用Camelot提取器
    camelot_extractor = ExtractorFactory.create('camelot')
    camelot_params = {
        'pdf_path': 'example.pdf',
        'page_num': 1,
        'flavor': 'lattice',  # 或 'stream'
        'param_mode': 'auto',  # 'default', 'auto', 'custom'
        'score_threshold': 0.5
    }
    camelot_results = camelot_extractor.extract_tables(page, feature_analyzer, camelot_params)
    
    # 使用PDFPlumber提取器
    pdfplumber_extractor = ExtractorFactory.create('pdfplumber')
    pdfplumber_params = {
        'flavor': 'lines',  # 或 'text'
        'param_mode': 'auto',
        'score_threshold': 0.5
    }
    pdfplumber_results = pdfplumber_extractor.extract_tables(page, feature_analyzer, pdfplumber_params)
```

### 2. 参数计算

```python
from core.extractors.factory import ExtractorFactory
from core.processing.page_feature_analyzer import PageFeatureAnalyzer
import pdfplumber

with pdfplumber.open('example.pdf') as pdf:
    page = pdf.pages[0]
    feature_analyzer = PageFeatureAnalyzer(page, enable_logging=False)
    
    # 获取Camelot参数
    camelot_extractor = ExtractorFactory.create('camelot')
    table_type = feature_analyzer.predict_table_type()  # 'bordered' 或 'unbordered'
    
    # 计算lattice参数
    lattice_params = camelot_extractor.calculate_params(
        feature_analyzer, 
        table_type='bordered',
        flavor='lattice',
        image_shape=(1200, 800)  # 可选，用于lattice模式
    )
    
    # 计算stream参数
    stream_params = camelot_extractor.calculate_params(
        feature_analyzer,
        table_type='unbordered',
        flavor='stream'
    )
    
    # 获取PDFPlumber参数
    pdfplumber_extractor = ExtractorFactory.create('pdfplumber')
    pdfplumber_params = pdfplumber_extractor.calculate_params(
        feature_analyzer,
        table_type='bordered'  # 或 'unbordered'
    )
```

## CamelotExtractor详细说明

### 支持的Flavor

- `lattice`: 适用于有边框的表格
- `stream`: 适用于无边框的表格

### 参数说明

#### Lattice模式参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `line_scale` | int | 40 | [15, 100] | 线条缩放因子，根据线条宽度动态调整 |
| `line_tol` | float | 2.0 | [0.5, 3.0] | 线条合并容差 |
| `joint_tol` | float | 2.0 | [0.5, 3.0] | 交点容差 |

#### Stream模式参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `edge_tol` | float | 50.0 | [动态] | 文本边缘容差，根据页面元素动态计算 |
| `row_tol` | float | 2.0 | [2, mode_height×1.5] | 行容差，基于字符高度众数 |
| `column_tol` | float | 0.0 | [0, 5.0] | 列容差 |

### 使用示例

```python
from core.extractors.camelot_extractor import CamelotExtractor
from core.processing.page_feature_analyzer import PageFeatureAnalyzer
import pdfplumber

# 创建提取器
extractor = CamelotExtractor()

# 打开PDF
with pdfplumber.open('example.pdf') as pdf:
    page = pdf.pages[0]
    feature_analyzer = PageFeatureAnalyzer(page, enable_logging=False)
    
    # 提取表格（lattice模式）
    params = {
        'pdf_path': 'example.pdf',
        'page_num': 1,
        'flavor': 'lattice',
        'param_mode': 'auto',
        'score_threshold': 0.6
    }
    results = extractor.extract_tables(page, feature_analyzer, params)
    
    # 处理结果
    for result in results:
        print(f"Score: {result['score']:.3f}")
        print(f"Source: {result['source']}")
        print(f"BBox: {result['bbox']}")
        # 获取表格DataFrame
        df = result['table'].df
        print(df)
```

## PDFPlumberExtractor详细说明

### 支持的Flavor

- `lines`: 适用于有边框的表格（基于线条检测）
- `text`: 适用于无边框的表格（基于文本对齐）

### 参数说明

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `snap_tolerance` | float | 2.0 | [0.5, 15] | 线条合并容差，根据字符尺寸动态调整 |
| `join_tolerance` | float | 2.0 | [1, 10] | 线段连接容差 |
| `edge_min_length` | float | 3.0 | [1, 30] | 边缘最小长度 |
| `intersection_tolerance` | float | 3.0 | [1, 10] | 交点容差 |
| `min_words_vertical` | int | 1 | [1, 10] | 垂直方向最小单词数，根据文本行数动态调整 |
| `min_words_horizontal` | int | 1 | [1, 5] | 水平方向最小单词数 |
| `text_x_tolerance` | float | 3.0 | [1, 30] | 文本X方向容差，根据字符尺寸动态调整 |
| `text_y_tolerance` | float | 5.0 | [1, 8] | 文本Y方向容差 |
| `vertical_strategy` | str | 'lines'/'text' | ['lines', 'text', 'explicit'] | 垂直策略，自动选择 |
| `horizontal_strategy` | str | 'lines'/'text' | ['lines', 'text', 'explicit'] | 水平策略，自动选择 |

### 使用示例

```python
from core.extractors.pdfplumber_extractor import PDFPlumberExtractor
from core.processing.page_feature_analyzer import PageFeatureAnalyzer
import pdfplumber

# 创建提取器
extractor = PDFPlumberExtractor()

# 打开PDF
with pdfplumber.open('example.pdf') as pdf:
    page = pdf.pages[0]
    feature_analyzer = PageFeatureAnalyzer(page, enable_logging=False)
    
    # 提取表格（lines模式）
    params = {
        'flavor': 'lines',
        'param_mode': 'auto',
        'score_threshold': 0.6
    }
    results = extractor.extract_tables(page, feature_analyzer, params)
    
    # 处理结果
    for result in results:
        print(f"Score: {result['score']:.3f}")
        print(f"Source: {result['source']}")
        # 获取表格DataFrame
        df = result['table'].df
        print(df)
```

## 参数模式说明

### 1. Default模式

使用库的默认参数：

```python
params = {
    'param_mode': 'default',
    'flavor': 'lattice'
}
```

### 2. Auto模式（推荐）

根据页面特征自动计算参数：

```python
params = {
    'param_mode': 'auto',
    'flavor': 'lattice'  # 可选，不指定则自动选择
}
```

### 3. Custom模式

使用自定义参数：

```python
params = {
    'param_mode': 'custom',
    'camelot_lattice_custom_params': {
        'line_scale': 50,
        'line_tol': 2.5,
        'joint_tol': 2.5
    }
}
```

## 结果格式

所有提取器返回的结果格式统一：

```python
[
    {
        'table': 表格对象（Camelot Table或PDFPlumberTableWrapper）,
        'bbox': [x0, y0, x1, y1],  # 边界框
        'score': 0.85,  # 评分 (0-1)
        'details': {...},  # 详细信息
        'domain': 'unstructured',  # 表格域类型
        'source': 'camelot_lattice'  # 来源标识
    },
    ...
]
```

### 访问表格数据

```python
for result in results:
    # 获取DataFrame
    df = result['table'].df
    
    # 获取边界框
    bbox = result['bbox']
    
    # 获取评分
    score = result['score']
```

## 工厂模式使用

### 列出可用的提取器

```python
from core.extractors.factory import ExtractorFactory

available = ExtractorFactory.list_available()
print(available)  # ['camelot', 'pdfplumber']
```

### 检查提取器是否注册

```python
if ExtractorFactory.is_registered('camelot'):
    extractor = ExtractorFactory.create('camelot')
```

### 注册自定义提取器

```python
from core.extractors.base import BaseExtractor
from core.extractors.factory import ExtractorFactory

class CustomExtractor(BaseExtractor):
    # 实现抽象方法
    ...

# 注册
ExtractorFactory.register('custom', CustomExtractor)

# 使用
extractor = ExtractorFactory.create('custom')
```

## 依赖关系

### CamelotExtractor依赖

- `camelot-py`: Camelot库
- `core.processing.page_feature_analyzer`: 页面特征分析
- `core.processing.table_params_calculator`: 参数计算
- `core.processing.table_evaluator`: 表格评估

### PDFPlumberExtractor依赖

- `pdfplumber`: PDFPlumber库
- `core.processing.page_feature_analyzer`: 页面特征分析
- `core.processing.table_params_calculator`: 参数计算
- `core.processing.table_evaluator`: 表格评估

## 最小化依赖配置

如果只需要使用提取器功能，最小依赖包括：

```txt
# 提取器核心依赖
pdfplumber>=0.9.0
camelot-py>=0.11.0
numpy>=1.23.0
pandas>=2.0.0
scipy>=1.10.0

# 工具依赖
Pillow>=10.0.0
```

## 常见问题

### Q: 如何单独使用提取器而不依赖整个项目？

A: 复制以下目录和文件到新项目：
- `core/extractors/` 目录
- `core/processing/page_feature_analyzer.py`
- `core/processing/table_params_calculator.py`
- `core/processing/table_type_classifier.py`
- `core/processing/table_evaluator.py`
- `core/utils/` 目录（工具函数）

### Q: 参数计算失败怎么办？

A: 使用default模式或提供custom参数：

```python
params = {
    'param_mode': 'default',  # 使用默认参数
    'flavor': 'lattice'
}
```

### Q: 如何自定义参数计算逻辑？

A: 继承BaseExtractor并重写`calculate_params`方法：

```python
class CustomCamelotExtractor(CamelotExtractor):
    def calculate_params(self, feature_analyzer, table_type: str, **kwargs) -> Dict:
        # 自定义参数计算逻辑
        params = super().calculate_params(feature_analyzer, table_type, **kwargs)
        # 修改参数
        params['line_scale'] = 50
        return params
```

## 完整示例

```python
"""
完整的表格提取示例
"""
from core.extractors.factory import ExtractorFactory
from core.processing.page_feature_analyzer import PageFeatureAnalyzer
import pdfplumber

def extract_tables_from_pdf(pdf_path: str, method: str = 'auto'):
    """
    从PDF提取表格
    
    Args:
        pdf_path: PDF文件路径
        method: 提取方法（'camelot', 'pdfplumber', 'auto'）
    """
    with pdfplumber.open(pdf_path) as pdf:
        all_tables = []
        
        for page_num, page in enumerate(pdf.pages, 1):
            print(f"Processing page {page_num}...")
            
            # 创建特征分析器
            feature_analyzer = PageFeatureAnalyzer(page, enable_logging=False)
            
            # 自动选择方法
            if method == 'auto':
                table_type = feature_analyzer.predict_table_type()
                if table_type == 'bordered':
                    method = 'camelot'
                    flavor = 'lattice'
                else:
                    method = 'pdfplumber'
                    flavor = 'text'
            
            # 创建提取器
            extractor = ExtractorFactory.create(method)
            
            # 准备参数
            params = {
                'flavor': flavor if method == 'camelot' else ('lines' if table_type == 'bordered' else 'text'),
                'param_mode': 'auto',
                'score_threshold': 0.5
            }
            
            if method == 'camelot':
                params['pdf_path'] = pdf_path
                params['page_num'] = page_num
            
            # 提取表格
            results = extractor.extract_tables(page, feature_analyzer, params)
            
            # 处理结果
            for result in results:
                all_tables.append({
                    'page': page_num,
                    'score': result['score'],
                    'source': result['source'],
                    'data': result['table'].df
                })
        
        return all_tables

# 使用示例
if __name__ == '__main__':
    tables = extract_tables_from_pdf('example.pdf', method='auto')
    for table in tables:
        print(f"Page {table['page']}, Score: {table['score']:.3f}")
        print(table['data'])
        print('-' * 50)
```

## 更多信息

- 参数计算原理：参考 [parameter_calculation_formulas.md](../parameter_calculation_formulas.md)
- 参数范围说明：参考 [parameter_range_documentation.md](../parameter_range_documentation.md)
- 移植指南：参考 [porting_guide.md](../porting_guide.md)
