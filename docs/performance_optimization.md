# 性能优化指南

本文档提供性能优化的建议和最佳实践。

## 引擎选择建议

### PDF文件处理

| 方法 | 速度 | 内存 | 适用场景 |
|------|------|------|----------|
| PDFPlumber | ⚡ 快 | 💾 低 | 无框表格、简单表格 |
| Camelot | 🐢 慢 | 💾 中 | 有框表格、复杂表格 |

**建议**：
- 优先使用PDFPlumber（通常更快）
- 只有在PDFPlumber效果不好时才使用Camelot
- 使用auto模式让系统自动选择

### 图像文件处理

| 引擎 | 速度 | 内存 | 中文支持 | 适用场景 |
|------|------|------|----------|----------|
| PaddleOCR | ⚡ 快 | 💾 中 | ✅ 优秀 | 中文文档、标准表格 |
| Transformer | 🐢 慢 | 💾 高 | ❌ 不支持 | 复杂表格、英文文档 |

**建议**：
- 中文文档优先使用PaddleOCR
- 复杂表格或英文文档使用Transformer
- 需要快速处理时使用PaddleOCR

## 参数优化

### 自动参数模式（推荐）

- 系统根据页面特征自动计算最优参数
- 适合大多数情况
- 无需手动调整

### 手动参数模式

如果自动参数效果不理想，可以尝试：

**PDFPlumber参数**：
- `snap_tolerance`：减小可提高精度，但可能漏检
- `min_words_vertical`：增大可减少误检
- `text_x_tolerance`：根据字符间距调整

**Camelot参数**：
- `line_scale`：根据线条粗细调整
- `row_tol`：根据行间距调整
- `edge_tol`：根据边缘清晰度调整

## 硬件加速

### GPU加速

**PaddleOCR**：
```python
# 在代码中启用GPU
engine = EngineFactory.create_detection('paddleocr', use_gpu=True)
```

**Transformer**：
```bash
# 确保CUDA正确安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**注意事项**：
- GPU加速需要NVIDIA GPU和CUDA
- 首次使用需要下载GPU版本的模型
- 内存占用会增加

### CPU优化

**MKLDNN加速（Intel CPU）**：
```python
# PaddleOCR启用MKLDNN
engine = EngineFactory.create_detection('paddleocr', enable_mkldnn=True)
```

**多线程处理**（计划中）：
- 批量处理时使用多线程
- 并行处理多个页面

## 批量处理优化

### 引擎实例复用

```python
# 只初始化一次，重复使用
engine = EngineFactory.create_detection('paddleocr')
engine.load_models()

# 处理多个文件
for file in files:
    tables = engine.detect_tables(file)
    # 处理结果...
```

### 分页处理

对于大文件，分页处理可以减少内存占用：

```python
# 一次处理一页
with pdfplumber.open('large.pdf') as pdf:
    for page in pdf.pages:
        # 处理单页
        results = extract_tables(page)
```

### 缓存机制

- 模型缓存：首次使用后模型会缓存，后续更快
- 结果缓存：可以缓存处理结果（如果文件未变化）

## 内存优化

### 减少内存占用

1. **处理较小文件**：
   - 一次处理一页
   - 分批处理大文件

2. **关闭不需要的引擎**：
   - 只加载使用的引擎
   - 使用懒加载机制

3. **使用CPU版本**：
   - 如果不需要GPU，使用CPU版本
   - CPU版本内存占用更小

4. **清理缓存**：
   ```python
   import gc
   gc.collect()  # 强制垃圾回收
   ```

### 监控内存使用

```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.2f} MB")
```

## 处理速度优化

### 预处理优化

1. **图像预处理**：
   - 调整图像大小（如果过大）
   - 转换为RGB格式
   - 去除噪声

2. **PDF预处理**：
   - 提取特定页面
   - 跳过空白页

### 并行处理（计划中）

```python
# 使用多进程处理多个文件
from multiprocessing import Pool

def process_file(file_path):
    # 处理单个文件
    pass

with Pool(processes=4) as pool:
    results = pool.map(process_file, file_list)
```

## 模型优化

### 模型选择

- **轻量级模型**：速度更快，精度略低
- **标准模型**：平衡速度和精度
- **高精度模型**：精度更高，速度较慢

### 模型缓存

模型会自动缓存，位置：
- Linux/Mac: `~/.cache/`
- Windows: `C:\Users\<username>\.cache\`

可以手动管理缓存：
```bash
# 清理缓存（如果需要）
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/paddleocr/
```

## 性能监控

### 处理时间统计

```python
import time

start_time = time.time()
# 处理操作
elapsed_time = time.time() - start_time
print(f"Processing time: {elapsed_time:.2f} seconds")
```

### 性能分析

使用Python性能分析工具：

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# 执行操作
process_file('example.pdf')

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # 显示前10个最耗时的函数
```

## 最佳实践

### 1. 选择合适的引擎

- 根据文档类型选择
- 根据精度要求选择
- 根据速度要求选择

### 2. 使用自动参数

- 优先使用自动参数模式
- 只有在必要时才手动调整

### 3. 批量处理优化

- 复用引擎实例
- 分页处理大文件
- 使用缓存机制

### 4. 硬件配置

- 使用GPU加速（如果可用）
- 使用SSD存储（提高I/O速度）
- 增加内存（处理大文件）

### 5. 监控和调优

- 监控处理时间
- 监控内存使用
- 根据实际情况调整配置

## 性能基准测试

### 测试环境

- CPU: Intel i7-9700K
- GPU: NVIDIA RTX 3080
- RAM: 32GB
- Python: 3.10

### 测试结果（示例）

| 方法/引擎 | 单页处理时间 | 内存占用 |
|-----------|--------------|----------|
| PDFPlumber | ~0.5s | ~50MB |
| Camelot | ~2.0s | ~100MB |
| PaddleOCR | ~1.0s | ~200MB |
| Transformer | ~5.0s | ~500MB |

**注意**：实际性能取决于文件复杂度、硬件配置等因素。

## 故障排除

### 处理速度慢

1. 检查硬件配置
2. 检查是否使用GPU
3. 检查文件大小和复杂度
4. 尝试使用更快的引擎

### 内存不足

1. 处理较小的文件
2. 使用CPU版本
3. 关闭不需要的引擎
4. 增加系统内存

### GPU未使用

1. 检查CUDA安装
2. 检查GPU驱动
3. 检查模型是否支持GPU
4. 检查代码配置

---

**提示**：性能优化是一个持续的过程，需要根据实际使用情况不断调整。
