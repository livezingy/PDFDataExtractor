# 模块移植指南

## 概述

本文档说明如何将提取器和引擎模块移植到其他项目中，实现代码复用和独立使用。

## 移植策略

### 方案1：完整移植（推荐）

复制整个模块目录和相关依赖，保持完整功能。

### 方案2：最小化移植

只复制必要的文件，减少依赖。

---

## 提取器模块移植

### 需要复制的文件

```
目标项目/
├── extractors/              # 复制整个目录
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── camelot_extractor.py
│   └── pdfplumber_extractor.py
├── processing/              # 复制相关处理模块
│   ├── page_feature_analyzer.py
│   ├── table_type_classifier.py
│   ├── table_params_calculator.py
│   └── table_evaluator.py
└── utils/                   # 复制工具模块
    ├── logger.py
    ├── param_config.py
    └── path_utils.py
```

### 依赖关系图

```
CamelotExtractor
    ├─ PageFeatureAnalyzer
    │   └─ pdfplumber
    ├─ TableParamsCalculator
    │   └─ PageFeatureAnalyzer
    └─ TableEvaluator
        └─ numpy, pandas, scipy

PDFPlumberExtractor
    ├─ PageFeatureAnalyzer
    │   └─ pdfplumber
    ├─ TableParamsCalculator
    │   └─ PageFeatureAnalyzer
    └─ TableEvaluator
        └─ numpy, pandas, scipy
```

### 最小依赖列表

```txt
# 核心依赖
pdfplumber>=0.9.0
camelot-py>=0.11.0

# 数据处理
numpy>=1.23.0
pandas>=2.0.0
scipy>=1.10.0

# 工具
Pillow>=10.0.0
```

### 移植步骤

1. **复制文件**
   ```bash
   # 复制extractors目录
   cp -r core/extractors/ your_project/extractors/
   
   # 复制processing目录中的相关文件
   cp core/processing/page_feature_analyzer.py your_project/processing/
   cp core/processing/table_type_classifier.py your_project/processing/
   cp core/processing/table_params_calculator.py your_project/processing/
   cp core/processing/table_evaluator.py your_project/processing/
   
   # 复制utils目录
   cp -r core/utils/ your_project/utils/
   ```

2. **修改导入路径**
   
   将所有`from core.`改为`from .`或根据你的项目结构调整：
   
   ```python
   # 原代码
   from core.extractors.base import BaseExtractor
   
   # 修改为（如果放在项目根目录）
   from extractors.base import BaseExtractor
   
   # 或（如果保持core结构）
   from core.extractors.base import BaseExtractor
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **测试使用**
   ```python
   from extractors.factory import ExtractorFactory
   # 使用提取器...
   ```

---

## OCR引擎模块移植

### EasyOCREngine移植

#### 需要复制的文件

```
目标项目/
├── engines/
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   └── easyocr_engine.py
└── utils/
    ├── logger.py
    ├── easyocr_config.py
    └── path_utils.py
```

#### 最小依赖

```txt
easyocr>=1.7.0
Pillow>=10.0.0
numpy>=1.23.0
```

#### 使用示例

```python
from engines.easyocr_engine import EasyOCREngine
from PIL import Image

engine = EasyOCREngine(languages=['en'])
engine.initialize()

image = Image.open('image.png')
results = engine.recognize_text(image)
```

### TransformerEngine移植

#### 需要复制的文件

```
目标项目/
├── engines/
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   └── transformer_engine.py
└── utils/
    └── logger.py
```

#### 最小依赖

```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
Pillow>=10.0.0
numpy>=1.23.0
```

#### 使用示例

```python
from engines.transformer_engine import TransformerEngine
from PIL import Image

engine = TransformerEngine(
    detection_model_path='microsoft/table-transformer-detection',
    structure_model_path='microsoft/table-transformer-structure-recognition'
)
engine.load_models()

image = Image.open('document.png')
tables = engine.detect_tables(image)
```

---

## 完整项目结构示例

### 示例1：仅使用提取器

```
my_table_extractor/
├── extractors/
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── camelot_extractor.py
│   └── pdfplumber_extractor.py
├── processing/
│   ├── __init__.py
│   ├── page_feature_analyzer.py
│   ├── table_type_classifier.py
│   ├── table_params_calculator.py
│   └── table_evaluator.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── param_config.py
│   └── path_utils.py
├── requirements.txt
└── main.py
```

### 示例2：仅使用OCR引擎

```
my_ocr_tool/
├── engines/
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   └── easyocr_engine.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── easyocr_config.py
│   └── path_utils.py
├── requirements.txt
└── main.py
```

### 示例3：完整功能

```
my_pdf_processor/
├── extractors/
│   └── ... (所有提取器文件)
├── engines/
│   └── ... (所有引擎文件)
├── processing/
│   └── ... (所有处理模块)
├── utils/
│   └── ... (所有工具模块)
├── requirements.txt
└── main.py
```

---

## 接口使用说明

### 提取器标准接口

所有提取器都实现以下接口：

```python
class BaseExtractor:
    def extract_tables(self, page, feature_analyzer, params: Dict) -> List[Dict]:
        """提取表格"""
        pass
    
    def calculate_params(self, feature_analyzer, table_type: str, **kwargs) -> Dict:
        """计算参数"""
        pass
    
    @property
    def name(self) -> str:
        """提取器名称"""
        pass
    
    @property
    def supported_flavors(self) -> List[str]:
        """支持的flavor列表"""
        pass
```

### OCR引擎标准接口

所有OCR引擎都实现以下接口：

```python
class BaseOCREngine:
    def recognize_text(self, image: Image.Image, **kwargs) -> List[Dict]:
        """识别文本"""
        pass
    
    @property
    def name(self) -> str:
        """引擎名称"""
        pass
```

### 检测引擎标准接口

所有检测引擎都实现以下接口：

```python
class BaseDetectionEngine:
    def detect_tables(self, image: Image.Image, **kwargs) -> List[Dict]:
        """检测表格"""
        pass
    
    def recognize_structure(self, image: Image.Image, table_bbox: Optional[List] = None, **kwargs) -> Dict:
        """识别表格结构"""
        pass
    
    @property
    def name(self) -> str:
        """引擎名称"""
        pass
```

---

## 移植检查清单

### 提取器移植

- [ ] 复制`extractors/`目录
- [ ] 复制`processing/`相关文件
- [ ] 复制`utils/`相关文件
- [ ] 修改所有导入路径
- [ ] 安装依赖包
- [ ] 测试基本功能
- [ ] 测试参数计算
- [ ] 测试表格提取

### OCR引擎移植

- [ ] 复制`engines/`目录
- [ ] 复制`utils/`相关文件
- [ ] 修改所有导入路径
- [ ] 安装依赖包
- [ ] 测试OCR功能
- [ ] 配置模型路径（如需要）

### 检测引擎移植

- [ ] 复制`engines/`目录
- [ ] 复制`utils/`相关文件
- [ ] 修改所有导入路径
- [ ] 安装依赖包
- [ ] 配置模型路径
- [ ] 测试检测功能
- [ ] 测试结构识别功能

---

## 常见移植问题

### Q1: 导入错误

**问题**：`ModuleNotFoundError: No module named 'core'`

**解决**：修改导入路径，根据你的项目结构调整：

```python
# 原代码
from core.extractors.base import BaseExtractor

# 方案1：如果放在项目根目录
from extractors.base import BaseExtractor

# 方案2：如果保持core结构，确保core在Python路径中
import sys
sys.path.insert(0, '/path/to/project')
from core.extractors.base import BaseExtractor
```

### Q2: 依赖缺失

**问题**：某些依赖包未安装

**解决**：安装所有必需的依赖：

```bash
pip install pdfplumber camelot-py numpy pandas scipy Pillow
```

### Q3: 配置文件路径错误

**问题**：找不到配置文件或模型路径

**解决**：检查`utils/path_utils.py`中的路径设置，或直接指定绝对路径：

```python
from engines.easyocr_engine import EasyOCREngine

engine = EasyOCREngine()
# 修改配置路径（如果需要）
engine._config.model_dir = '/path/to/models'
```

### Q4: 日志系统错误

**问题**：日志系统初始化失败

**解决**：简化日志系统或使用标准logging：

```python
# 如果logger.py有问题，可以简化
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

---

## 独立使用示例

### 示例：独立的表格提取工具

```python
"""
独立的表格提取工具
不依赖整个PDFDataExtractor项目
"""
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(__file__))

from extractors.factory import ExtractorFactory
from processing.page_feature_analyzer import PageFeatureAnalyzer
import pdfplumber

def extract_tables(pdf_path: str, method: str = 'auto'):
    """提取PDF中的表格"""
    with pdfplumber.open(pdf_path) as pdf:
        all_tables = []
        
        for page_num, page in enumerate(pdf.pages, 1):
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
                'flavor': flavor,
                'param_mode': 'auto',
                'score_threshold': 0.5
            }
            
            if method == 'camelot':
                params['pdf_path'] = pdf_path
                params['page_num'] = page_num
            
            # 提取表格
            results = extractor.extract_tables(page, feature_analyzer, params)
            
            # 收集结果
            for result in results:
                all_tables.append({
                    'page': page_num,
                    'score': result['score'],
                    'source': result['source'],
                    'data': result['table'].df
                })
        
        return all_tables

if __name__ == '__main__':
    tables = extract_tables('example.pdf')
    for table in tables:
        print(f"Page {table['page']}, Score: {table['score']:.3f}")
        print(table['data'])
```

---

## 版本兼容性

### 提取器模块

- **Python版本**：>= 3.8
- **pdfplumber**：>= 0.9.0
- **camelot-py**：>= 0.11.0

### OCR引擎模块

- **Python版本**：>= 3.8
- **easyocr**：>= 1.7.0
- **torch**：>= 2.0.0（仅Transformer需要）

---

## 获取帮助

- 查看提取器使用文档：[extractors/usage.md](extractors/usage.md)
- 查看引擎使用文档：[engines/usage.md](engines/usage.md)
- 查看参数计算原理：[parameter_calculation_formulas.md](parameter_calculation_formulas.md)

---

**文档版本**：v1.0  
**创建日期**：2025-12-12  
**最后更新**：2025-12-12
