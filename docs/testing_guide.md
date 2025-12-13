# 测试指南

## 概述

本文档说明如何运行和编写项目的测试。

## 测试结构

项目使用pytest作为测试框架，测试文件位于`tests/`目录。

## 快速开始

### 1. 安装测试依赖

```bash
pip install -r requirements-test.txt
```

### 2. 运行所有测试

```bash
pytest
```

### 3. 运行特定测试

```bash
# 运行提取器测试
pytest tests/extractors/

# 运行引擎测试
pytest tests/engines/

# 运行特定测试文件
pytest tests/extractors/test_camelot_extractor.py
```

## 测试覆盖率

### 生成覆盖率报告

```bash
pytest --cov=core --cov-report=html
```

覆盖率报告将生成在`htmlcov/index.html`，可以在浏览器中打开查看。

### 查看终端覆盖率报告

```bash
pytest --cov=core --cov-report=term-missing
```

## 测试标记

项目使用pytest标记来分类测试：

- `@pytest.mark.unit`: 单元测试
- `@pytest.mark.integration`: 集成测试
- `@pytest.mark.slow`: 慢速测试
- `@pytest.mark.requires_camelot`: 需要Camelot库
- `@pytest.mark.requires_easyocr`: 需要EasyOCR库
- `@pytest.mark.requires_transformer`: 需要Transformer库

### 运行特定标记的测试

```bash
# 只运行单元测试
pytest -m unit

# 跳过慢速测试
pytest -m "not slow"

# 运行需要特定库的测试
pytest -m requires_camelot
```

## 编写测试

### 测试提取器

```python
import pytest
from unittest.mock import Mock, patch
from core.extractors.factory import ExtractorFactory

class TestMyExtractor:
    def test_basic_functionality(self, mock_page, mock_feature_analyzer):
        extractor = ExtractorFactory.create('camelot')
        
        params = {
            'pdf_path': 'test.pdf',
            'page_num': 1,
            'flavor': 'lattice',
            'param_mode': 'auto'
        }
        
        # 使用mock进行测试
        with patch.object(extractor, '_extract_lattice') as mock_extract:
            mock_extract.return_value = []
            results = extractor.extract_tables(mock_page, mock_feature_analyzer, params)
            assert results == []
```

### 测试引擎

#### EasyOCR引擎测试

```python
import pytest
from unittest.mock import Mock, patch
from core.engines.factory import EngineFactory
from PIL import Image

class TestEasyOCREngine:
    def test_recognize_text(self, sample_image):
        engine = EngineFactory.create_ocr('easyocr')
        
        with patch.object(engine, '_reader') as mock_reader:
            mock_reader.readtext.return_value = [
                ([[10, 10], [100, 10], [100, 30], [10, 30]], 'Text', 0.95)
            ]
            engine._initialized = True
            
            results = engine.recognize_text(sample_image)
            assert len(results) > 0
```

#### PaddleOCR引擎测试

```python
import pytest
from core.engines.factory import EngineFactory
from PIL import Image
import numpy as np

class TestPaddleOCREngine:
    def test_ocr_recognition(self, sample_image):
        """测试PaddleOCR文本识别"""
        engine = EngineFactory.create_ocr('paddleocr', lang='ch', use_gpu=False)
        
        if not engine.initialize():
            pytest.skip("PaddleOCR not available")
        
        results = engine.recognize_text(sample_image)
        assert isinstance(results, list)
        if results:
            assert 'text' in results[0]
            assert 'confidence' in results[0]
    
    def test_table_detection(self, sample_image):
        """测试PaddleOCR表格检测"""
        engine = EngineFactory.create_detection('paddleocr', use_gpu=False)
        
        if not engine.load_models():
            pytest.skip("PaddleOCR PP-Structure not available")
        
        tables = engine.detect_tables(sample_image)
        assert isinstance(tables, list)
        if tables:
            assert 'bbox' in tables[0]
            assert 'confidence' in tables[0]
    
    def test_structure_recognition(self, sample_image):
        """测试PaddleOCR结构识别"""
        engine = EngineFactory.create_detection('paddleocr', use_gpu=False)
        
        if not engine.load_models():
            pytest.skip("PaddleOCR PP-Structure not available")
        
        structure = engine.recognize_structure(sample_image, return_raw=True)
        assert isinstance(structure, dict)
```

#### Transformer引擎测试

```python
import pytest
from core.engines.factory import EngineFactory
from PIL import Image

class TestTransformerEngine:
    def test_table_detection(self, sample_image):
        """测试Transformer表格检测"""
        engine = EngineFactory.create_detection('transformer')
        
        if not engine.load_models():
            pytest.skip("Transformer models not available")
        
        tables = engine.detect_tables(sample_image)
        assert isinstance(tables, list)
    
    def test_structure_recognition(self, sample_image):
        """测试Transformer结构识别"""
        engine = EngineFactory.create_detection('transformer')
        
        if not engine.load_models():
            pytest.skip("Transformer models not available")
        
        structure = engine.recognize_structure(sample_image)
        assert isinstance(structure, dict)
```

## Streamlit界面测试

### 测试Streamlit组件

```python
import pytest
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.streamlit_utils import check_dependencies, save_uploaded_file
from unittest.mock import Mock, patch

class TestStreamlitComponents:
    def test_sidebar_rendering(self):
        """测试侧边栏渲染"""
        # 注意：需要Streamlit运行时环境
        # 实际测试可能需要使用streamlit.testing
        pass
    
    def test_dependency_check(self):
        """测试依赖检查"""
        dependencies = check_dependencies()
        assert isinstance(dependencies, dict)
        assert 'pdfplumber' in dependencies
        assert 'camelot' in dependencies
        assert 'paddleocr' in dependencies
        assert 'transformer' in dependencies
    
    def test_file_processing(self, tmp_path):
        """测试文件处理"""
        from streamlit_app.streamlit_utils import process_pdf_streamlit
        
        # 创建测试PDF文件
        test_pdf = tmp_path / "test.pdf"
        # ... 创建测试PDF ...
        
        # 测试处理
        # results = process_pdf_streamlit(...)
        # assert results is not None
```

### 测试引擎选择功能

```python
import pytest
from streamlit_app.streamlit_utils import (
    _process_image_with_paddleocr,
    _process_image_with_transformer
)
from PIL import Image
import numpy as np

class TestEngineSelection:
    def test_paddleocr_processing(self):
        """测试PaddleOCR图像处理"""
        # 创建测试图像
        image = Image.new('RGB', (800, 600), color='white')
        
        params = {}
        results = {'detection_steps': []}
        
        try:
            tables = _process_image_with_paddleocr(image, params, results)
            assert isinstance(tables, list)
        except ImportError:
            pytest.skip("PaddleOCR not available")
    
    def test_transformer_processing(self):
        """测试Transformer图像处理"""
        # 创建测试图像
        image = Image.new('RGB', (800, 600), color='white')
        
        params = {}
        results = {'detection_steps': []}
        
        try:
            tables = _process_image_with_transformer(image, params, results)
            assert isinstance(tables, list)
        except ImportError:
            pytest.skip("Transformer not available")
```

## 性能优化测试

### 测试模块加载性能

```python
import time
import sys

def test_module_import_performance():
    """测试模块导入性能"""
    start = time.time()
    
    # 清除模块缓存
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('core.extractors')]
    for k in modules_to_remove:
        del sys.modules[k]
    
    # 重新导入
    from core.extractors import ExtractorFactory
    
    elapsed = time.time() - start
    assert elapsed < 0.1  # 应该在100ms内完成

def test_engine_lazy_loading():
    """测试引擎懒加载性能"""
    import time
    
    # 测试懒加载
    start = time.time()
    from core.engines.factory import EngineFactory
    
    # 不实际创建引擎，只测试导入时间
    elapsed = time.time() - start
    assert elapsed < 0.5  # 导入应该很快
```

### 测试处理性能

```python
import pytest
import time
from core.extractors.factory import ExtractorFactory
import pdfplumber

def test_extraction_performance(sample_pdf_path):
    """测试提取性能"""
    with pdfplumber.open(sample_pdf_path) as pdf:
        page = pdf.pages[0]
        
        extractor = ExtractorFactory.create('pdfplumber')
        
        start = time.time()
        # 执行提取
        # results = extractor.extract_tables(...)
        elapsed = time.time() - start
        
        # 性能断言（根据实际情况调整）
        assert elapsed < 10.0  # 应该在10秒内完成
```

### 测试懒加载

```python
def test_lazy_loading():
    """测试懒加载机制"""
    # 导入模块不应该立即加载所有提取器
    from core.extractors import ExtractorFactory
    
    # 检查提取器是否已注册（懒加载应该已经触发）
    assert ExtractorFactory.is_registered('camelot')
```

## 持续集成

### GitHub Actions示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run tests
        run: pytest --cov=core --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## 最佳实践

1. **使用Fixtures**：利用`conftest.py`中的fixtures避免重复代码
2. **模拟外部依赖**：使用`unittest.mock`模拟外部库
3. **测试隔离**：每个测试应该是独立的
4. **清晰的测试名称**：使用描述性的测试名称
5. **测试文档**：为复杂测试添加文档字符串

## 故障排除

### 测试失败：模块未找到

确保项目根目录在Python路径中：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### 测试失败：依赖缺失

确保安装了所有测试依赖：

```bash
pip install -r requirements-test.txt
```

### 测试失败：权限问题

某些测试可能需要文件系统权限，确保测试目录可写。

---

**文档版本**：v1.0  
**创建日期**：2025-12-12
