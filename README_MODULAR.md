# 模块化架构说明

## 概述

本项目已完成模块化重构，将Camelot、PDFPlumber、EasyOCR、Transformer、PaddleOCR等功能模块化为独立的可移植插件。项目已删除GUI界面，专注于Streamlit Web界面。

## 新架构

### 目录结构

```
core/
├── extractors/              # 表格提取器模块
│   ├── __init__.py         # 自动注册提取器
│   ├── base.py             # BaseExtractor基类
│   ├── factory.py          # ExtractorFactory工厂类
│   ├── camelot_extractor.py
│   ├── pdfplumber_extractor.py
│   └── requirements.txt    # 提取器最小依赖
├── engines/                 # OCR/检测引擎模块
│   ├── __init__.py         # 自动注册引擎
│   ├── base.py             # BaseOCREngine和BaseDetectionEngine基类
│   ├── factory.py          # EngineFactory工厂类
│   ├── easyocr_engine.py
│   ├── transformer_engine.py
│   └── requirements.txt    # 引擎最小依赖
└── processing/              # 处理模块（保留）
    ├── page_feature_analyzer.py
    ├── table_type_classifier.py
    ├── table_params_calculator.py
    └── table_processor.py  # 已更新使用新接口
```

## 快速使用

### 使用提取器

```python
from core.extractors.factory import ExtractorFactory
from core.processing.page_feature_analyzer import PageFeatureAnalyzer
import pdfplumber

# 打开PDF
with pdfplumber.open('example.pdf') as pdf:
    page = pdf.pages[0]
    feature_analyzer = PageFeatureAnalyzer(page, enable_logging=False)
    
    # 使用Camelot提取器
    camelot_extractor = ExtractorFactory.create('camelot')
    results = camelot_extractor.extract_tables(
        page, 
        feature_analyzer,
        {
            'pdf_path': 'example.pdf',
            'page_num': 1,
            'flavor': 'lattice',
            'param_mode': 'auto',
            'score_threshold': 0.5
        }
    )
```

### 使用OCR引擎

```python
from core.engines.factory import EngineFactory
from PIL import Image

# 创建OCR引擎（EasyOCR）
ocr_engine = EngineFactory.create_ocr('easyocr', languages=['en'])
ocr_engine.initialize()

# 识别文本
image = Image.open('table.png')
results = ocr_engine.recognize_text(image, min_confidence=0.5)
```

### 使用PaddleOCR引擎（新增）

```python
from core.engines.factory import EngineFactory
from PIL import Image

# 创建PaddleOCR引擎（同时支持OCR和表格检测）
ocr_engine = EngineFactory.create_ocr('paddleocr', lang='ch', use_gpu=False)
ocr_engine.initialize()

# OCR识别
image = Image.open('table.png')
text_results = ocr_engine.recognize_text(image)

# 表格检测和结构识别
detection_engine = EngineFactory.create_detection('paddleocr')
detection_engine.load_models()
tables = detection_engine.detect_tables(image)
for table in tables:
    structure = detection_engine.recognize_structure(image.crop(table['bbox']))
```

## 模块独立性

### 提取器模块

- **可独立使用**：复制`core/extractors/`和相关依赖即可
- **标准接口**：所有提取器实现`BaseExtractor`接口
- **工厂模式**：通过`ExtractorFactory`统一管理

### OCR引擎模块

- **可独立使用**：复制`core/engines/`和相关依赖即可
- **标准接口**：所有引擎实现`BaseOCREngine`或`BaseDetectionEngine`接口
- **工厂模式**：通过`EngineFactory`统一管理

## 文档

- [提取器使用文档](docs/extractors/usage.md)
- [引擎使用文档](docs/engines/usage.md)
- [PaddleOCR使用文档](docs/engines/paddleocr_usage.md)
- [移植指南](docs/porting_guide.md)

## 向后兼容性

- `TableProcessor`的公共接口保持不变
- 旧的提取方法（`extract_camelot_lattice`等）保留但已标记为deprecated
- 新代码使用新的提取器接口

## 主要改进

1. **模块化**：每个功能模块独立，可单独使用
2. **标准化**：统一的接口设计，易于扩展
3. **可移植**：模块可轻松移植到其他项目
4. **工厂模式**：统一的创建和管理方式
5. **PaddleOCR集成**：新增PaddleOCR引擎，支持OCR和表格检测（PP-Structure）
6. **Streamlit界面**：现代化的Web界面，替代原有GUI
7. **引擎选择**：图像文件可选择PaddleOCR或Transformer引擎
8. **智能界面**：根据文件类型自动切换选择选项

## 版本信息

- **当前版本**：v2.0.0
- **最后更新**：2025-12-12
- **变更日志**：查看 [CHANGELOG.md](../CHANGELOG.md)
