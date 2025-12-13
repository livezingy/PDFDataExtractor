# Streamlit Cloud 部署指南

本文档说明如何在Streamlit Cloud上部署PDF Table Extractor，提供Camelot、PDFPlumber和PaddleOCR的一键试用功能。

## 概述

Streamlit Cloud提供免费的云端部署服务，适合快速演示和试用。本项目的以下功能可以在Streamlit Cloud上使用：

- ✅ **PDFPlumber**：PDF表格提取
- ✅ **Camelot**：PDF表格提取（有框表格）
- ✅ **PaddleOCR**：图像表格检测和识别

⚠️ **Transformer**：由于资源限制，仅在本地部署可用

## 快速部署

### 步骤1：准备GitHub仓库

1. 确保代码已推送到GitHub
2. 确保包含以下文件：
   - `requirements_streamlit.txt`：Python依赖
   - `packages.txt`：系统依赖（可选）
   - `.streamlit/config.toml`：Streamlit配置
   - `streamlit_app/streamlit_app.py`：主应用文件

### 步骤2：部署到Streamlit Cloud

1. 访问 [Streamlit Cloud](https://streamlit.io/cloud)
2. 使用GitHub账户登录
3. 点击"New app"
4. 选择仓库：`livezingy/PDFDataExtractor`
5. 配置部署：
   - **Main file path**: `streamlit_app/streamlit_app.py`
   - **Python version**: 3.10（或更高）
   - **Branch**: `main`（或你的主分支）
6. 点击"Deploy"

### 步骤3：等待部署完成

- 首次部署可能需要5-10分钟
- Streamlit Cloud会自动安装依赖
- 部署完成后会提供公开URL

## 配置文件说明

### requirements_streamlit.txt

包含Streamlit Cloud部署所需的所有Python依赖：

```txt
# PDF处理
pdfplumber>=0.9.0
camelot-py>=0.11.0

# PaddleOCR
paddleocr>=2.7.0
paddlepaddle>=2.5.0

# 其他依赖...
```

**注意**：不包含Transformer相关依赖（torch、transformers等），因为Streamlit Cloud免费版资源有限。

### packages.txt

包含系统级依赖（通过apt-get安装）：

```txt
ghostscript  # Camelot依赖
```

### .streamlit/config.toml

Streamlit配置文件：

```toml
[server]
maxUploadSize = 10  # MB，测试版本限制
fileWatcherType = "none"  # 避免事件循环错误
```

## 功能说明

### 可用功能

#### 1. PDF文件处理

- **PDFPlumber**：✅ 完全可用
  - 支持auto、lines、text模式
  - 自动参数计算
  - 适合无框表格

- **Camelot**：✅ 完全可用
  - 支持auto、lattice、stream模式
  - 自动参数计算
  - 适合有框表格

#### 2. 图像文件处理

- **PaddleOCR**：✅ 完全可用
  - 表格检测
  - 结构识别
  - HTML输出
  - 中文识别优秀

### 不可用功能

- **Transformer**：❌ 仅在本地环境可用
  - 原因：模型文件较大（500MB+），内存需求高
  - 替代方案：使用PaddleOCR
  - 本地部署：参考 [部署指南](deployment_guide.md)

## 环境检测

应用会自动检测是否在Streamlit Cloud环境：

- 检测环境变量：`STREAMLIT_CLOUD`、`STREAMLIT_SHARING`
- 检测路径：`/home/appuser`（Streamlit Cloud默认用户目录）
- 自动调整功能可用性

## 使用限制

### Streamlit Cloud免费版限制

- **内存**：约1GB
- **CPU**：共享资源
- **存储**：约1GB
- **文件上传**：最大200MB（配置为10MB用于测试）
- **运行时间**：无限制（但资源有限）

### 本项目的限制

- **文件大小**：10MB（测试版本）
- **Transformer**：不可用
- **GPU加速**：不支持

## 优化建议

### 1. 模型加载优化

PaddleOCR会在首次使用时下载模型，建议：

- 显示加载进度
- 缓存模型文件
- 提供加载状态提示

### 2. 内存优化

- 处理较小的文件
- 一次处理一页
- 及时释放资源

### 3. 用户体验

- 清晰的错误提示
- 功能可用性说明
- 本地部署指引

## 故障排除

### 问题1：部署失败

**可能原因**：
- 依赖安装失败
- 系统依赖缺失
- 配置文件错误

**解决方案**：
1. 检查`requirements_streamlit.txt`格式
2. 检查`packages.txt`（如果使用）
3. 查看Streamlit Cloud日志

### 问题2：Camelot不可用

**可能原因**：
- Ghostscript未安装
- 系统库缺失

**解决方案**：
1. 确保`packages.txt`包含`ghostscript`
2. 检查部署日志
3. 应用会自动降级处理

### 问题3：PaddleOCR加载慢

**说明**：首次使用需要下载模型，这是正常的。

**优化**：
- 等待首次加载完成
- 后续使用会更快（模型已缓存）

### 问题4：内存不足

**可能原因**：
- 文件过大
- 同时处理多个文件

**解决方案**：
- 使用较小的测试文件
- 一次处理一个文件
- 考虑升级到付费版

## 升级到付费版

如果需要更多资源（如Transformer支持），可以考虑：

1. **Streamlit Cloud付费版**：
   - 更多内存和CPU
   - 可能支持Transformer
   - 查看 [Streamlit Cloud定价](https://streamlit.io/cloud)

2. **自建服务器**：
   - 完全控制资源
   - 支持所有功能
   - 参考 [部署指南](deployment_guide.md)

## 本地部署对比

| 功能 | Streamlit Cloud | 本地部署 |
|------|----------------|----------|
| PDFPlumber | ✅ | ✅ |
| Camelot | ✅ | ✅ |
| PaddleOCR | ✅ | ✅ |
| Transformer | ❌ | ✅ |
| GPU加速 | ❌ | ✅ |
| 资源限制 | 有 | 无 |
| 成本 | 免费 | 服务器成本 |

## 获取帮助

- 查看 [部署指南](deployment_guide.md) 了解本地部署
- 查看 [FAQ](FAQ.md) 解决常见问题
- 提交 [GitHub Issue](https://github.com/livezingy/PDFDataExtractor/issues)

---

**提示**：Streamlit Cloud适合快速试用和演示。对于生产环境或需要Transformer功能，建议使用本地部署。
