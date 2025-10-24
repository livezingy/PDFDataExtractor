# EasyOCR本地模型配置说明

## 概述

本项目已配置EasyOCR使用本地模型，优先使用 `models/EasyOCR/model/` 目录下的模型文件，如果本地模型不存在则自动下载。

## 目录结构

```
PDFDataExtractor/
├── models/
│   └── EasyOCR/
│       └── model/
│           ├── craft_mlt_25k.pth      # CRAFT文本检测模型
│           ├── english_g2.pth         # 英文识别模型
│           └── cache/                 # 缓存目录（自动创建）
```

## 配置特性

### 1. 自动模型检测
- 系统启动时自动检查本地模型是否存在
- 如果模型存在，使用本地模型（`download_enabled=False`）
- 如果模型不存在，自动下载（`download_enabled=True`）

### 2. 环境变量设置
- `EASYOCR_MODULE_PATH`: 设置为 `models/EasyOCR/model/`
- `EASYOCR_CACHE_DIR`: 设置为 `models/EasyOCR/model/cache/`

### 3. 多语言支持
支持以下语言的本地模型：
- `en`: 英文
- `ch_sim`: 简体中文
- `ch_tra`: 繁体中文
- `ja`: 日文
- `ko`: 韩文
- `th`: 泰文
- `vi`: 越南文
- `ar`: 阿拉伯文
- `hi`: 印地文

## 使用方法

### 基本使用

```python
from core.utils.easyocr_config import get_easyocr_reader

# 创建英文OCR Reader
reader = get_easyocr_reader(['en'])

# 执行OCR
results = reader.readtext(image)
```

### 多语言使用

```python
# 支持多语言
reader = get_easyocr_reader(['en', 'ch_sim'])

# 使用GPU（如果可用）
reader = get_easyocr_reader(['en'], gpu=True)
```

### 高级配置

```python
from core.utils.easyocr_config import get_easyocr_config

# 获取配置实例
config = get_easyocr_config()

# 手动下载模型
success = config.download_models(['en', 'ch_sim'])

# 获取模型信息
model_info = config.get_model_info()
print(f"CRAFT模型存在: {model_info['craft_model']['exists']}")
print(f"英文模型存在: {model_info['recognition_models']['en']['exists']}")
```

## 项目中的集成

### 1. 表格解析器 (table_parser.py)
```python
# 原来的代码
reader = easyocr.Reader(['en'], download_enabled=False)

# 现在的代码
reader = get_easyocr_reader(['en'])
```

### 2. 表格模型 (table_models.py)
```python
# 导入配置
from core.utils.easyocr_config import get_easyocr_reader

# 使用配置的Reader
reader = get_easyocr_reader(['en'])
```

## 测试验证

运行测试脚本验证配置：

```bash
python test_easyocr_config.py
```

测试内容包括：
1. 配置初始化
2. 模型文件检查
3. Reader创建
4. OCR功能测试
5. 模型下载测试

## 优势

### 1. 离线使用
- 本地模型存在时完全离线工作
- 无需网络连接即可进行OCR

### 2. 性能优化
- 避免重复下载模型
- 减少网络请求时间
- 提高启动速度

### 3. 版本控制
- 模型文件版本固定
- 避免自动更新导致的兼容性问题
- 便于项目部署

### 4. 存储管理
- 集中管理所有模型文件
- 便于备份和迁移
- 减少磁盘空间占用

## 故障排除

### 1. 模型文件损坏
如果本地模型文件损坏，删除对应文件后重新运行程序，系统会自动下载：

```bash
# 删除损坏的模型文件
rm models/EasyOCR/model/craft_mlt_25k.pth
rm models/EasyOCR/model/english_g2.pth

# 重新运行程序，会自动下载
python main.py
```

### 2. 网络问题
如果网络连接有问题，确保本地模型文件完整：

```python
from core.utils.easyocr_config import get_easyocr_config

config = get_easyocr_config()
model_info = config.get_model_info()

# 检查模型状态
print("模型状态:", model_info)
```

### 3. 权限问题
确保程序对模型目录有读写权限：

```bash
# Windows
icacls models\EasyOCR\model /grant Everyone:F

# Linux/Mac
chmod -R 755 models/EasyOCR/model/
```

## 配置参数

### 环境变量
- `EASYOCR_MODULE_PATH`: 模型存储路径
- `EASYOCR_CACHE_DIR`: 缓存目录路径

### 配置选项
- `languages`: 支持的语言列表
- `gpu`: 是否使用GPU加速
- `download_enabled`: 是否允许下载模型

## 更新日志

- **v1.0**: 初始版本，支持本地模型配置
- **v1.1**: 添加多语言支持
- **v1.2**: 优化模型检测逻辑
- **v1.3**: 添加测试脚本和文档

## 注意事项

1. **模型文件大小**: 每个模型文件约100-200MB，请确保有足够磁盘空间
2. **首次运行**: 首次运行时会自动下载模型，需要网络连接
3. **版本兼容**: 确保EasyOCR版本与模型文件版本兼容
4. **GPU支持**: 如需GPU加速，请确保CUDA环境正确配置

## 技术支持

如有问题，请检查：
1. 模型文件是否完整
2. 网络连接是否正常
3. 权限设置是否正确
4. 日志文件中的错误信息

更多信息请参考EasyOCR官方文档：https://github.com/JaidedAI/EasyOCR
