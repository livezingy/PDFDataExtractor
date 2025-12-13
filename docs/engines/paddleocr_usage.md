# PaddleOCR引擎使用文档

## 概述

PaddleOCR引擎提供了完整的OCR和表格检测功能，包括：
- **文本识别**：支持中英文等多种语言的OCR
- **表格检测**：使用PP-Structure进行表格检测
- **表格结构识别**：识别表格的单元格、行列结构

## 快速开始

### 1. 基本OCR使用

```python
from core.engines.factory import EngineFactory
from PIL import Image

# 创建OCR引擎
ocr_engine = EngineFactory.create_ocr('paddleocr', lang='ch', use_gpu=False)

# 初始化引擎
ocr_engine.initialize()

# 读取图像
image = Image.open('table.png')

# 识别文本
results = ocr_engine.recognize_text(image)

# 处理结果
for result in results:
    print(f"Text: {result['text']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"BBox: {result['bbox']}")
```

### 2. 表格检测使用

```python
from core.engines.factory import EngineFactory
from PIL import Image

# 创建检测引擎
detection_engine = EngineFactory.create_detection('paddleocr', use_gpu=False)

# 加载模型
detection_engine.load_models()

# 读取图像
image = Image.open('document.png')

# 检测表格
tables = detection_engine.detect_tables(image)

# 处理检测结果
for table in tables:
    print(f"Table found with confidence {table['confidence']:.3f}")
    print(f"BBox: {table['bbox']}")
```

### 3. 表格结构识别

```python
from core.engines.factory import EngineFactory
from PIL import Image

# 创建检测引擎
engine = EngineFactory.create_detection('paddleocr', use_gpu=False)
engine.load_models()

# 读取图像
image = Image.open('table.png')

# 识别表格结构
structure = engine.recognize_structure(image, return_raw=True)

# 获取HTML格式的表格
if structure.get('html'):
    print("Table HTML:")
    print(structure['html'])

# 获取单元格信息
if structure.get('cells'):
    print(f"Found {len(structure['cells'])} cells")
```

## 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_angle_cls` | bool | True | 是否使用角度分类器 |
| `lang` | str | 'ch' | 语言，'ch'（中文）或'en'（英文） |
| `use_gpu` | bool | False | 是否使用GPU |
| `enable_mkldnn` | bool | False | 是否启用MKLDNN加速（CPU优化） |
| `table_model_dir` | str | None | 表格模型目录（PP-Structure） |

## 方法说明

### recognize_text()

识别图像中的文本。

**参数**：
- `image`: PIL Image对象
- `det`: 是否进行文本检测（默认True）
- `rec`: 是否进行文本识别（默认True）
- `cls`: 是否进行角度分类（默认True）

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

### detect_tables()

检测图像中的表格区域。

**参数**：
- `image`: PIL Image对象
- `**kwargs`: 其他参数（保留用于未来扩展）

**注意**：PPStructure的`__call__`方法不接受`layout`参数，直接传入图像数组即可。

**返回**：
```python
[
    {
        'bbox': [x1, y1, x2, y2],
        'confidence': 0.95,
        'type': 'table',
        'raw': {...}  # 原始数据
    },
    ...
]
```

### recognize_structure()

识别表格结构。

**参数**：
- `image`: PIL Image对象（表格区域）
- `table_bbox`: 表格边界框 [x1, y1, x2, y2]（可选）
- `return_raw`: 是否返回原始输出（默认False）

**返回**：
```python
{
    'html': '<table>...</table>',  # HTML格式的表格
    'cells': [...],  # 单元格列表
    'rows': 5,  # 行数
    'columns': 4,  # 列数
    'raw': {...}  # 原始数据（如果return_raw=True）
}
```

## 使用示例

### 完整示例：OCR + 表格检测

```python
"""
使用PaddleOCR进行OCR和表格检测的完整示例
"""
from core.engines.paddleocr_engine import PaddleOCREngine
from PIL import Image

# 创建引擎
engine = PaddleOCREngine(lang='ch', use_gpu=False)

# 初始化OCR
if not engine.initialize():
    print("Failed to initialize PaddleOCR")
    exit(1)

# 加载表格检测模型
if not engine.load_models():
    print("Failed to load PP-Structure models")
    exit(1)

# 读取图像
image = Image.open('document.png')

# 1. 进行OCR
ocr_results = engine.recognize_text(image)
print(f"Found {len(ocr_results)} text regions")

# 2. 检测表格
tables = engine.detect_tables(image)
print(f"Found {len(tables)} tables")

# 3. 识别每个表格的结构
for i, table in enumerate(tables):
    bbox = table['bbox']
    print(f"\nTable {i+1}:")
    print(f"  BBox: {bbox}")
    print(f"  Confidence: {table['confidence']:.3f}")
    
    # 裁剪表格区域
    table_image = image.crop(bbox)
    
    # 识别结构
    structure = engine.recognize_structure(table_image)
    
    if structure.get('html'):
        print(f"  HTML rows: {structure.get('rows', 'N/A')}")
        print(f"  HTML columns: {structure.get('columns', 'N/A')}")
    
    if structure.get('cells'):
        print(f"  Cells: {len(structure['cells'])}")
```

### 中英文混合识别

```python
from core.engines.paddleocr_engine import PaddleOCREngine
from PIL import Image

# 创建引擎（中文模式，也支持英文）
engine = PaddleOCREngine(lang='ch', use_gpu=False)
engine.initialize()

image = Image.open('mixed_language.png')
results = engine.recognize_text(image)

for result in results:
    print(f"{result['text']} (confidence: {result['confidence']:.3f})")
```

### 批量处理

```python
from core.engines.paddleocr_engine import PaddleOCREngine
from PIL import Image
import os

# 创建引擎（只初始化一次）
engine = PaddleOCREngine(lang='ch', use_gpu=False)
engine.initialize()
engine.load_models()

# 批量处理图像
image_dir = 'images/'
for filename in os.listdir(image_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        
        # OCR
        ocr_results = engine.recognize_text(image)
        print(f"{filename}: {len(ocr_results)} text regions")
        
        # 表格检测
        tables = engine.detect_tables(image)
        print(f"{filename}: {len(tables)} tables")
```

## 性能优化

### 1. GPU加速

```python
# 使用GPU（如果可用）
engine = PaddleOCREngine(lang='ch', use_gpu=True)
```

### 2. CPU优化（MKLDNN）

```python
# 启用MKLDNN加速（Intel CPU优化）
engine = PaddleOCREngine(lang='ch', enable_mkldnn=True)
```

### 3. 只进行文本识别（跳过检测）

```python
# 如果已经知道文本位置，可以跳过检测步骤
results = engine.recognize_text(image, det=False, rec=True)
```

## 依赖关系

### PaddleOCREngine依赖

- `paddleocr>=2.7.0`: PaddleOCR库
- `paddlepaddle>=2.5.0`: PaddlePaddle深度学习框架
- `Pillow>=10.0.0`: 图像处理
- `numpy>=1.23.0`: 数值计算

## 安装

```bash
# 安装PaddleOCR
pip install paddleocr paddlepaddle

# 如果使用GPU
pip install paddlepaddle-gpu
```

## 常见问题

### Q: PaddleOCR初始化很慢怎么办？

A: 这是正常的，PaddleOCR需要加载模型。建议：
1. 只初始化一次，重复使用引擎实例
2. 使用GPU加速
3. 如果不需要角度分类，设置`use_angle_cls=False`

### Q: 如何指定模型路径？

A: 可以通过环境变量或代码指定：

```python
# 通过代码指定表格模型路径
engine = PaddleOCREngine(table_model_dir='/path/to/models')
```

### Q: 支持哪些语言？

A: PaddleOCR支持多种语言，常见的有：
- `ch`: 中文
- `en`: 英文
- `korean`: 韩文
- `japan`: 日文
- 等等

### Q: 如何获取表格的HTML输出？

A: 使用`recognize_structure()`方法：

```python
structure = engine.recognize_structure(image)
html = structure.get('html', '')
```

## 与EasyOCR和Transformer的对比

| 特性 | PaddleOCR | EasyOCR | Transformer |
|------|-----------|---------|-------------|
| OCR支持 | ✅ 优秀 | ✅ 良好 | ❌ 不支持 |
| 中文识别 | ✅ 优秀 | ✅ 良好 | ❌ 不支持 |
| 表格检测 | ✅ PP-Structure | ❌ 不支持 | ✅ 支持 |
| 表格结构识别 | ✅ 支持 | ❌ 不支持 | ✅ 支持 |
| 模型大小 | 中等 | 较大 | 较大 |
| 速度 | 快 | 中等 | 慢 |

## 更多信息

- PaddleOCR官方文档：https://github.com/PaddlePaddle/PaddleOCR
- PP-Structure文档：https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/README_ch.md
- 移植指南：参考 [porting_guide.md](../porting_guide.md)
