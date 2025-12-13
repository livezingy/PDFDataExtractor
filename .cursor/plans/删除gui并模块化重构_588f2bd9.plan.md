---
name: 删除GUI并模块化重构
overview: 删除GUI部分，仅保留Streamlit；将Camelot、PDFPlumber、EasyOCR、Transformer模块化为独立可移植的插件，采用基类+工厂模式，提供标准接口和使用文档。
todos:
  - id: delete_gui
    content: 删除GUI部分：删除gui/目录、main.py，更新requirements.txt移除PySide6，清理相关引用
    status: completed
  - id: create_base_structure
    content: 创建基础架构：创建extractors/和engines/目录，实现BaseExtractor和BaseOCREngine基类，实现工厂类
    status: completed
  - id: modularize_camelot
    content: 模块化Camelot：创建CamelotExtractor类，迁移参数计算和提取逻辑，注册到工厂，更新TableProcessor
    status: completed
  - id: modularize_pdfplumber
    content: 模块化PDFPlumber：创建PDFPlumberExtractor类，迁移参数计算和提取逻辑，注册到工厂，更新TableProcessor
    status: completed
  - id: modularize_easyocr
    content: 模块化EasyOCR：创建EasyOCREngine类，迁移配置和初始化逻辑，实现标准OCR接口
    status: completed
  - id: modularize_transformer
    content: 模块化Transformer：创建TransformerEngine类，迁移模型加载逻辑，实现检测和识别接口
    status: completed
  - id: create_documentation
    content: 创建文档：编写提取器和引擎使用文档，创建移植指南，编写示例代码
    status: completed
  - id: testing_validation
    content: 测试和验证：编写单元测试，进行集成测试，验证模块可移植性
    status: completed
---

# 删除GUI并模块化重构计划

## 一、删除GUI部分

### 1.1 删除GUI相关文件和目录

- **删除目录**：`gui/`（包含所有GUI相关代码）
- **删除文件**：`main.py`（GUI入口文件）
- **更新依赖**：`requirements.txt`（移除PySide6）

### 1.2 更新Streamlit入口

- **文件**：`streamlit_app/streamlit_app.py`
- **操作**：确保Streamlit应用作为唯一入口点
- **验证**：确保所有功能在Streamlit中正常工作

### 1.3 清理相关引用

- **搜索并删除**：所有对`gui/`目录的导入
- **更新文档**：README.md、CHANGELOG.md

---

## 二、模块化重构架构设计

### 2.1 新的目录结构

```
core/
├── extractors/              # 表格提取器模块（新增）
│   ├── __init__.py
│   ├── base.py             # BaseExtractor基类
│   ├── factory.py          # ExtractorFactory工厂类
│   ├── camelot_extractor.py
│   └── pdfplumber_extractor.py
├── engines/                 # OCR/检测引擎模块（新增）
│   ├── __init__.py
│   ├── base.py             # BaseOCREngine基类
│   ├── factory.py          # EngineFactory工厂类
│   ├── easyocr_engine.py
│   └── transformer_engine.py
├── processing/              # 保留现有处理模块
│   ├── page_feature_analyzer.py
│   ├── table_type_classifier.py
│   ├── table_params_calculator.py
│   └── table_processor.py  # 重构，使用新的extractors
└── utils/                  # 工具模块（保留）
```

### 2.2 接口设计

#### 2.2.1 BaseExtractor基类（`core/extractors/base.py`）

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pdfplumber

class BaseExtractor(ABC):
    """表格提取器基类"""
    
    @abstractmethod
    def extract_tables(self, page, feature_analyzer, params: Dict) -> List[Dict]:
        """
        提取表格
        
        Args:
            page: pdfplumber.Page对象
            feature_analyzer: PageFeatureAnalyzer实例
            params: 参数字典
            
        Returns:
            List[Dict]: 表格结果列表，每个元素包含：
                - table: 表格对象
                - bbox: 边界框
                - score: 评分
                - source: 来源标识
        """
        pass
    
    @abstractmethod
    def calculate_params(self, feature_analyzer, table_type: str) -> Dict:
        """
        计算参数
        
        Args:
            feature_analyzer: PageFeatureAnalyzer实例
            table_type: 'bordered' 或 'unbordered'
            
        Returns:
            Dict: 参数字典
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """提取器名称"""
        pass
    
    @property
    @abstractmethod
    def supported_flavors(self) -> List[str]:
        """支持的flavor列表"""
        pass
```

#### 2.2.2 BaseOCREngine基类（`core/engines/base.py`）

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from PIL import Image

class BaseOCREngine(ABC):
    """OCR引擎基类"""
    
    @abstractmethod
    def recognize_text(self, image: Image.Image, **kwargs) -> List[Dict]:
        """
        识别文本
        
        Args:
            image: PIL Image对象
            **kwargs: 其他参数
            
        Returns:
            List[Dict]: OCR结果列表，每个元素包含：
                - text: 文本内容
                - bbox: 边界框
                - confidence: 置信度
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""
        pass
```

---

## 三、具体模块化实现

### 3.1 Camelot提取器模块化

#### 3.1.1 创建CamelotExtractor（`core/extractors/camelot_extractor.py`）

**职责**：

- 封装Camelot的lattice和stream模式
- 集成参数计算逻辑（从`table_params_calculator.py`迁移）
- 提供统一的提取接口

**关键方法**：

- `extract_tables()`: 统一提取接口
- `extract_lattice()`: lattice模式提取
- `extract_stream()`: stream模式提取
- `calculate_lattice_params()`: 计算lattice参数
- `calculate_stream_params()`: 计算stream参数

**依赖关系**：

- 依赖：`PageFeatureAnalyzer`, `TableParamsCalculator`
- 被依赖：`TableProcessor`

### 3.2 PDFPlumber提取器模块化

#### 3.2.1 创建PDFPlumberExtractor（`core/extractors/pdfplumber_extractor.py`）

**职责**：

- 封装PDFPlumber的lines和text模式
- 集成参数计算逻辑
- 提供统一的提取接口

**关键方法**：

- `extract_tables()`: 统一提取接口
- `extract_lines()`: lines模式提取
- `extract_text()`: text模式提取
- `calculate_params()`: 计算参数（区分bordered/unbordered）

### 3.3 EasyOCR引擎模块化

#### 3.3.1 创建EasyOCREngine（`core/engines/easyocr_engine.py`）

**职责**：

- 封装EasyOCR的初始化和调用
- 提供标准OCR接口
- 处理模型加载和配置

**关键方法**：

- `recognize_text()`: 文本识别
- `initialize()`: 初始化引擎
- `get_reader()`: 获取EasyOCR reader实例

**配置管理**：

- 从`core/utils/easyocr_config.py`迁移配置逻辑
- 支持本地模型路径配置

### 3.4 Transformer引擎模块化

#### 3.4.1 创建TransformerEngine（`core/engines/transformer_engine.py`）

**职责**：

- 封装Transformer模型的加载和推理
- 提供表格检测和结构识别接口
- 处理图像预处理和后处理

**关键方法**：

- `detect_tables()`: 表格检测
- `recognize_structure()`: 结构识别
- `load_models()`: 加载模型
- `preprocess_image()`: 图像预处理

**依赖关系**：

- 从`core/models/table_models.py`迁移相关逻辑
- 保持与现有TableParser的兼容性

---

## 四、工厂模式实现

### 4.1 ExtractorFactory（`core/extractors/factory.py`）

```python
class ExtractorFactory:
    """提取器工厂类"""
    
    _extractors = {}
    
    @classmethod
    def register(cls, name: str, extractor_class):
        """注册提取器"""
        cls._extractors[name.lower()] = extractor_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseExtractor:
        """创建提取器实例"""
        name_lower = name.lower()
        if name_lower not in cls._extractors:
            raise ValueError(f"Unknown extractor: {name}")
        return cls._extractors[name_lower](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出可用的提取器"""
        return list(cls._extractors.keys())
```

### 4.2 EngineFactory（`core/engines/factory.py`）

类似实现，用于管理OCR引擎。

---

## 五、重构TableProcessor

### 5.1 更新TableProcessor（`core/processing/table_processor.py`）

**变更**：

- 移除直接的Camelot和PDFPlumber调用代码
- 使用ExtractorFactory创建提取器
- 简化提取逻辑，委托给相应的提取器

**关键变更**：

```python
# 旧代码
def extract_camelot_lattice(self, ...):
    # 直接调用camelot

# 新代码
def extract_camelot_lattice(self, ...):
    extractor = ExtractorFactory.create('camelot')
    return extractor.extract_tables(page, feature_analyzer, params)
```

---

## 六、创建使用文档和示例

### 6.1 提取器使用文档（`docs/extractors/usage.md`）

**内容**：

- 如何单独使用CamelotExtractor
- 如何单独使用PDFPlumberExtractor
- 参数计算接口说明
- 完整使用示例

### 6.2 OCR引擎使用文档（`docs/engines/usage.md`）

**内容**：

- 如何单独使用EasyOCREngine
- 如何单独使用TransformerEngine
- 配置说明
- 完整使用示例

### 6.3 移植指南（`docs/porting_guide.md`）

**内容**：

- 如何将提取器移植到其他项目
- 依赖关系说明
- 最小化依赖配置
- 示例项目结构

---

## 七、代码优化和清理

### 7.1 参数计算逻辑迁移

**从**：`core/processing/table_params_calculator.py`

**到**：

- `core/extractors/camelot_extractor.py`（Camelot参数计算）
- `core/extractors/pdfplumber_extractor.py`（PDFPlumber参数计算）

**保留**：`table_params_calculator.py`作为通用参数计算工具（可选）

### 7.2 依赖管理

**创建**：`core/extractors/requirements.txt`（提取器最小依赖）

**创建**：`core/engines/requirements.txt`（引擎最小依赖）

---

## 八、测试和验证

### 8.1 单元测试

- 为每个提取器创建测试
- 为每个引擎创建测试
- 测试工厂模式

### 8.2 集成测试

- 测试提取器在TableProcessor中的集成
- 测试Streamlit应用功能完整性

### 8.3 移植测试

- 创建独立的测试项目
- 验证模块的可移植性

---

## 九、实施步骤

### 阶段1：删除GUI（1-2天）

1. 删除`gui/`目录和`main.py`
2. 更新`requirements.txt`
3. 清理相关引用
4. 测试Streamlit功能

### 阶段2：创建基础架构（2-3天）

1. 创建`extractors/`和`engines/`目录
2. 实现基类（BaseExtractor, BaseOCREngine）
3. 实现工厂类（ExtractorFactory, EngineFactory）
4. 创建`__init__.py`文件

### 阶段3：模块化Camelot（2-3天）

1. 创建`CamelotExtractor`类
2. 迁移参数计算逻辑
3. 迁移提取逻辑
4. 注册到工厂
5. 更新TableProcessor使用新接口

### 阶段4：模块化PDFPlumber（2-3天）

1. 创建`PDFPlumberExtractor`类
2. 迁移参数计算逻辑
3. 迁移提取逻辑
4. 注册到工厂
5. 更新TableProcessor使用新接口

### 阶段5：模块化EasyOCR（1-2天）

1. 创建`EasyOCREngine`类
2. 迁移配置和初始化逻辑
3. 实现标准OCR接口
4. 注册到工厂

### 阶段6：模块化Transformer（2-3天）

1. 创建`TransformerEngine`类
2. 迁移模型加载逻辑
3. 实现检测和识别接口
4. 注册到工厂

### 阶段7：文档和测试（2-3天）

1. 编写使用文档
2. 编写移植指南
3. 创建示例代码
4. 编写单元测试

### 阶段8：清理和优化（1-2天）

1. 代码审查
2. 性能优化
3. 文档完善
4. 最终测试

---

## 十、关键文件清单

### 需要创建的文件：

- `core/extractors/__init__.py`
- `core/extractors/base.py`
- `core/extractors/factory.py`
- `core/extractors/camelot_extractor.py`
- `core/extractors/pdfplumber_extractor.py`
- `core/engines/__init__.py`
- `core/engines/base.py`
- `core/engines/factory.py`
- `core/engines/easyocr_engine.py`
- `core/engines/transformer_engine.py`
- `docs/extractors/usage.md`
- `docs/engines/usage.md`
- `docs/porting_guide.md`

### 需要修改的文件：

- `core/processing/table_processor.py`（重构使用新接口）
- `core/models/table_parser.py`（使用新的引擎接口）
- `requirements.txt`（移除PySide6）
- `README.md`（更新说明）

### 需要删除的文件：

- `gui/`目录（所有文件）
- `main.py`

---

## 十一、向后兼容性

### 11.1 保持现有API

- `PageFeatureAnalyzer`作为适配器保留
- `TableProcessor`的公共接口保持不变
- Streamlit应用无需修改

### 11.2 渐进式迁移

- 新代码使用新接口
- 旧代码继续工作
- 逐步迁移内部实现

---

## 十二、成功标准

- [ ] GUI完全删除，Streamlit正常工作
- [ ] 所有提取器模块化完成
- [ ] 所有引擎模块化完成
- [ ] 工厂模式正常工作
- [ ] 使用文档完整
- [ ] 移植指南清晰
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 代码可独立移植