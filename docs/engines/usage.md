# OCR/检测引擎使用文档

## 概述

本文档说明如何单独使用EasyOCR和Transformer引擎模块。这些模块已经模块化，可以独立使用或移植到其他项目中。

## 目录结构

```
core/engines/
├── __init__.py          # 模块初始化，自动注册引擎
├── base.py              # BaseOCREngine和BaseDetectionEngine基类
├── factory.py           # EngineFactory工厂类
├── easyocr_engine.py    # EasyOCR引擎
└── transformer_engine.py # Transformer引擎
```

## 快速开始

### 1. EasyOCR引擎基本使用

```python
from core.engines.factory import EngineFactory
from PIL import Image

# 创建OCR引擎
ocr_engine = EngineFactory.create_ocr('easyocr', languages=['en'], gpu=False)

# 初始化引擎
ocr_engine.initialize()

# 读取图像
image = Image.open('table.png')

# 识别文本
results = ocr_engine.recognize_text(image, min_confidence=0.5)

# 处理结果
for result in results:
    print(f"Text: {result['text']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"BBox: {result['bbox']}")
```

### 2. Transformer引擎基本使用

```python
from core.engines.factory import EngineFactory
from PIL import Image

# 创建检测引擎
detection_engine = EngineFactory.create_detection(
    'transformer',
    detection_model_path='path/to/detection/model',
    structure_model_path='path/to/structure/model',
    device='cpu'
)

# 加载模型
detection_engine.load_models()

# 读取图像
image = Image.open('document.png')

# 检测表格
tables = detection_engine.detect_tables(image, confidence_threshold=0.5)

# 识别表格结构
for table in tables:
    bbox = table['bbox']
    # 裁剪表格区域
    table_image = image.crop(bbox)
    # 识别结构
    structure = detection_engine.recognize_structure(table_image)
```

## EasyOCREngine详细说明

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `languages` | List[str] | ['en'] | 支持的语言列表 |
| `gpu` | bool | False | 是否使用GPU |

### 方法说明

#### recognize_text()

识别图像中的文本。

**参数**：
- `image`: PIL Image对象
- `min_confidence`: 最小置信度阈值（默认0.0）

**返回**：
```python
[
    {
        'text': '文本内容',
        'bbox': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # 角点格式
        'bbox_rect': [x1, y1, x2, y2],  # 矩形格式
        'confidence': 0.95
    },
    ...
]
```

#### recognize_text_in_region()

识别指定区域的文本。

**参数**：
- `image`: PIL Image对象
- `bbox`: 区域边界框 [x1, y1, x2, y2]
- `min_confidence`: 最小置信度阈值

**返回**：与`recognize_text()`相同格式，但坐标已调整到原图坐标系

### 使用示例

```python
from core.engines.easyocr_engine import EasyOCREngine
from PIL import Image

# 创建引擎
engine = EasyOCREngine(languages=['en', 'ch_sim'], gpu=False)

# 初始化
if not engine.initialize():
    print("Failed to initialize EasyOCR")
    exit(1)

# 读取图像
image = Image.open('table.png')

# 识别整个图像
all_text = engine.recognize_text(image, min_confidence=0.5)

# 识别指定区域
region_text = engine.recognize_text_in_region(
    image, 
    bbox=[100, 100, 500, 300],
    min_confidence=0.5
)

# 获取reader实例（高级用法）
reader = engine.get_reader()
# 可以直接使用reader进行更复杂的操作
```

## TransformerEngine详细说明

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `detection_model_path` | str | None | 检测模型路径或HuggingFace模型ID |
| `structure_model_path` | str | None | 结构识别模型路径或HuggingFace模型ID |
| `device` | str | 'cpu' | 设备（'cpu'或'cuda'） |
| `detection_confidence` | float | 0.5 | 检测置信度阈值 |
| `structure_confidence` | float | 0.5 | 结构识别置信度阈值 |

### 方法说明

#### detect_tables()

检测图像中的表格区域。

**参数**：
- `image`: PIL Image对象
- `confidence_threshold`: 置信度阈值（覆盖初始化时的设置）

**返回**：
```python
[
    {
        'bbox': [x1, y1, x2, y2],
        'confidence': 0.95,
        'label': 1
    },
    ...
]
```

#### recognize_structure()

识别表格结构。

**参数**：
- `image`: PIL Image对象（表格区域）
- `table_bbox`: 表格边界框 [x1, y1, x2, y2]（可选）
- `return_raw_outputs`: 是否返回原始输出（默认False）

**返回**：
- 如果`return_raw_outputs=False`：返回包含基本信息的字典
- 如果`return_raw_outputs=True`：返回包含模型、原始输出等的字典，用于高级处理

### 使用示例

```python
from core.engines.transformer_engine import TransformerEngine
from PIL import Image

# 创建引擎
engine = TransformerEngine(
    detection_model_path='microsoft/table-transformer-detection',
    structure_model_path='microsoft/table-transformer-structure-recognition',
    device='cpu',
    detection_confidence=0.5
)

# 加载模型
if not engine.load_models():
    print("Failed to load models")
    exit(1)

# 读取图像
image = Image.open('document.png')

# 检测表格
tables = engine.detect_tables(image, confidence_threshold=0.5)

# 处理每个表格
for table in tables:
    bbox = table['bbox']
    print(f"Found table with confidence {table['confidence']:.3f}")
    
    # 裁剪表格区域
    table_image = image.crop(bbox)
    
    # 识别结构（返回原始输出用于高级处理）
    structure = engine.recognize_structure(
        table_image,
        return_raw_outputs=True
    )
    
    # 使用原始输出进行后处理
    model = structure['model']
    outputs = structure['outputs']
    processor = structure['processor']
    image_size = structure['image_size']
    
    # 进行后处理（具体实现取决于需求）
    # ...
```

## 工厂模式使用

### 列出可用的引擎

```python
from core.engines.factory import EngineFactory

# 列出OCR引擎
ocr_engines = EngineFactory.list_available_ocr()
print(ocr_engines)  # ['easyocr']

# 列出检测引擎
detection_engines = EngineFactory.list_available_detection()
print(detection_engines)  # ['transformer']
```

### 检查引擎是否注册

```python
if EngineFactory.is_ocr_registered('easyocr'):
    engine = EngineFactory.create_ocr('easyocr')

if EngineFactory.is_detection_registered('transformer'):
    engine = EngineFactory.create_detection('transformer')
```

### 注册自定义引擎

```python
from core.engines.base import BaseOCREngine
from core.engines.factory import EngineFactory

class CustomOCREngine(BaseOCREngine):
    # 实现抽象方法
    ...

# 注册
EngineFactory.register_ocr('custom', CustomOCREngine)

# 使用
engine = EngineFactory.create_ocr('custom')
```

## 依赖关系

### EasyOCREngine依赖

- `easyocr>=1.7.0`: EasyOCR库
- `core.utils.easyocr_config`: EasyOCR配置管理
- `Pillow>=10.0.0`: 图像处理
- `numpy>=1.23.0`: 数值计算

### TransformerEngine依赖

- `torch>=2.0.0`: PyTorch
- `transformers>=4.30.0`: HuggingFace Transformers
- `torchvision>=0.15.0`: TorchVision
- `Pillow>=10.0.0`: 图像处理
- `numpy>=1.23.0`: 数值计算

## 最小化依赖配置

### EasyOCR引擎最小依赖

```txt
easyocr>=1.7.0
Pillow>=10.0.0
numpy>=1.23.0
```

### Transformer引擎最小依赖

```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
Pillow>=10.0.0
numpy>=1.23.0
```

## 常见问题

### Q: EasyOCR初始化失败怎么办？

A: 检查模型文件是否存在，或允许自动下载：

```python
engine = EasyOCREngine(languages=['en'])
# 确保模型目录存在或允许下载
engine.initialize()
```

### Q: Transformer模型加载失败怎么办？

A: 检查模型路径，或使用HuggingFace模型ID：

```python
# 使用HuggingFace模型ID（自动下载）
engine = TransformerEngine(
    detection_model_path='microsoft/table-transformer-detection',
    structure_model_path='microsoft/table-transformer-structure-recognition'
)
```

### Q: 如何单独使用引擎而不依赖整个项目？

A: 复制以下目录和文件到新项目：
- `core/engines/` 目录
- `core/utils/easyocr_config.py`（仅EasyOCR需要）
- `core/utils/logger.py`
- `core/utils/path_utils.py`（仅EasyOCR需要）

## 完整示例

### EasyOCR完整示例

```python
"""
使用EasyOCR进行表格OCR的完整示例
"""
from core.engines.easyocr_engine import EasyOCREngine
from PIL import Image
import pandas as pd

def ocr_table_image(image_path: str):
    """对表格图像进行OCR"""
    # 创建引擎
    engine = EasyOCREngine(languages=['en'], gpu=False)
    
    # 初始化
    if not engine.initialize():
        print("Failed to initialize EasyOCR")
        return None
    
    # 读取图像
    image = Image.open(image_path)
    
    # 识别文本
    results = engine.recognize_text(image, min_confidence=0.5)
    
    # 转换为DataFrame格式
    data = []
    for result in results:
        data.append({
            'text': result['text'],
            'x1': result['bbox_rect'][0],
            'y1': result['bbox_rect'][1],
            'x2': result['bbox_rect'][2],
            'y2': result['bbox_rect'][3],
            'confidence': result['confidence']
        })
    
    df = pd.DataFrame(data)
    return df

# 使用
df = ocr_table_image('table.png')
print(df)
```

### Transformer完整示例

```python
"""
使用Transformer进行表格检测和结构识别的完整示例
"""
from core.engines.transformer_engine import TransformerEngine
from PIL import Image

def detect_and_recognize_tables(image_path: str):
    """检测并识别表格"""
    # 创建引擎
    engine = TransformerEngine(
        detection_model_path='microsoft/table-transformer-detection',
        structure_model_path='microsoft/table-transformer-structure-recognition',
        device='cpu'
    )
    
    # 加载模型
    if not engine.load_models():
        print("Failed to load models")
        return []
    
    # 读取图像
    image = Image.open(image_path)
    
    # 检测表格
    tables = engine.detect_tables(image, confidence_threshold=0.5)
    
    results = []
    for table in tables:
        bbox = table['bbox']
        confidence = table['confidence']
        
        # 裁剪表格区域
        table_image = image.crop(bbox)
        
        # 识别结构
        structure = engine.recognize_structure(
            table_image,
            return_raw_outputs=True
        )
        
        results.append({
            'bbox': bbox,
            'confidence': confidence,
            'structure': structure
        })
    
    return results

# 使用
results = detect_and_recognize_tables('document.png')
for result in results:
    print(f"Table found with confidence {result['confidence']:.3f}")
    print(f"BBox: {result['bbox']}")
```

## PaddleOCR引擎

PaddleOCR提供了完整的OCR和表格检测功能。详细使用说明请参考：
- [PaddleOCR使用文档](paddleocr_usage.md)

### 快速示例

```python
from core.engines.factory import EngineFactory
from PIL import Image

# 创建PaddleOCR引擎（同时支持OCR和检测）
ocr_engine = EngineFactory.create_ocr('paddleocr', lang='ch')
ocr_engine.initialize()

detection_engine = EngineFactory.create_detection('paddleocr')
detection_engine.load_models()

# OCR
image = Image.open('table.png')
text_results = ocr_engine.recognize_text(image)

# 表格检测
tables = detection_engine.detect_tables(image)

# 表格结构识别
for table in tables:
    structure = detection_engine.recognize_structure(image.crop(table['bbox']))
```

## 更多信息

- 移植指南：参考 [porting_guide.md](../porting_guide.md)
- EasyOCR配置：参考 [easyocr_local_models_setup.md](../easyocr_local_models_setup.md)
- PaddleOCR使用：参考 [paddleocr_usage.md](paddleocr_usage.md)
