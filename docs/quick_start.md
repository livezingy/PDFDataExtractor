# 快速开始指南

## 5分钟快速上手

本指南将帮助你在5分钟内开始使用PDF Table Extractor。

## 方式选择

### 方式1：Streamlit Cloud一键试用（推荐）⭐

**无需安装，直接在浏览器中使用**

1. 访问项目的Streamlit Cloud部署（如果已部署）
2. 或参考 [Streamlit Cloud部署指南](streamlit_cloud_deployment.md) 自行部署
3. 支持功能：PDFPlumber、Camelot、PaddleOCR

**优点**：无需安装，快速试用  
**限制**：不支持Transformer功能

### 方式2：本地安装（完整功能）

**支持所有功能，包括Transformer**

## 前置要求

- Python 3.8 或更高版本
- pip 包管理器

## 步骤1：安装依赖

```bash
# 克隆或下载项目
cd PDFDataExtractor

# 安装核心依赖
pip install -r requirements.txt

# 安装Streamlit界面依赖
pip install -r requirements_streamlit.txt
```

**注意**：首次安装可能需要几分钟时间，因为需要下载一些深度学习模型。

## 步骤2：启动应用

```bash
streamlit run streamlit_app/streamlit_app.py
```

应用启动后，浏览器会自动打开（默认地址：http://localhost:8501）

## 步骤3：提取第一个PDF表格

### 3.1 准备测试文件

使用项目中的测试文件或准备一个包含表格的PDF文件。

### 3.2 在界面中操作

1. **上传文件**：在左侧边栏点击"Select PDF or Image File"，选择PDF文件
2. **选择方法**：选择"Extraction Method"（推荐使用PDFPlumber）
3. **选择Flavor**：选择"auto"（自动选择最佳模式）
4. **开始提取**：点击"🚀 Start Extraction"按钮
5. **查看结果**：在右侧查看提取的表格数据

### 3.3 结果说明

- **表格列表**：显示检测到的所有表格
- **表格数据**：以DataFrame格式显示表格内容
- **提取信息**：显示表格位置、置信度等信息

## 步骤4：提取第一个图像表格

### 4.1 准备图像文件

准备一个包含表格的图像文件（PNG、JPG等格式）。

### 4.2 在界面中操作

1. **上传文件**：选择图像文件
2. **选择引擎**：选择"Detection Engine"（推荐使用PaddleOCR）
3. **开始提取**：点击"🚀 Start Extraction"按钮
4. **查看结果**：查看检测和识别的表格

### 4.3 引擎选择建议

- **PaddleOCR**（推荐）：
  - 适合中文文档
  - 处理速度快
  - 支持HTML输出
  
- **Transformer**：
  - 适合复杂表格
  - 精度较高
  - 处理速度较慢

## 常见问题快速解决

### Q1: 启动时出现依赖错误

**解决方案**：
```bash
# 确保Python版本正确
python --version  # 应该是3.8+

# 重新安装依赖
pip install --upgrade -r requirements.txt
```

### Q2: 文件上传后无法处理

**检查项**：
- 文件大小是否超过10MB（测试版本限制）
- 文件格式是否支持（PDF或图像格式）
- 文件是否损坏

### Q3: 提取结果为空

**可能原因**：
- PDF中没有表格
- 表格类型与选择的方法不匹配
- 参数设置不当

**解决方案**：
- 尝试切换提取方法（PDFPlumber ↔ Camelot）
- 尝试不同的Flavor选项
- 检查PDF文件质量

### Q4: PaddleOCR初始化慢

**说明**：首次使用PaddleOCR时需要下载模型，这是正常现象。

**优化建议**：
- 等待首次初始化完成
- 后续使用会更快（模型已缓存）
- 考虑使用GPU加速（如果可用）

## 下一步

- 阅读 [Streamlit用户使用指南](streamlit_user_guide.md) 了解详细功能
- 查看 [常见问题FAQ](FAQ.md) 解决更多问题
- 参考 [技术原理文档](../README.md#-技术原理文档) 深入了解算法

## 获取帮助

- 查看 [GitHub Issues](https://github.com/livezingy/PDFDataExtractor/issues)
- 阅读完整文档：[文档索引](../README.md#-文档索引)

---

**提示**：如果遇到问题，请先查看 [FAQ](FAQ.md) 和 [常见问题快速解决](#常见问题快速解决) 部分。
