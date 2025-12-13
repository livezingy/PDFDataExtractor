# 📊 PDF Table Extractor

一个强大的PDF和图像表格提取工具，支持多种提取方法和OCR引擎，提供智能参数自动计算和表格类型识别功能。

## ✨ 主要特性

- **多种提取方法**：支持 PDFPlumber 和 Camelot 两种PDF表格提取方法
- **OCR引擎支持**：集成 EasyOCR、Transformer 和 PaddleOCR 三种OCR/检测引擎
- **智能参数计算**：基于页面特征自动计算最优提取参数
- **表格类型识别**：自动识别有框表格和无框表格
- **Streamlit界面**：现代化的Web界面，易于使用
- **模块化架构**：提取器和引擎模块化设计，易于扩展和移植

## 🚀 快速开始

### 方式1：Streamlit Cloud一键试用（推荐）⭐

**无需安装，直接在浏览器中使用**

1. 访问 [Streamlit Cloud部署](https://share.streamlit.io/) 或查看 [部署指南](docs/deployment_guide.md)
2. 支持功能：
   - ✅ PDFPlumber（PDF表格提取）
   - ✅ Camelot（PDF表格提取）
   - ✅ PaddleOCR（图像表格检测）
   - ❌ Transformer（仅本地部署可用）

### 方式2：本地安装

#### 环境要求

- Python >= 3.8
- 操作系统：Windows / Linux / macOS

#### 安装

```bash
# 克隆仓库
git clone https://github.com/livezingy/PDFDataExtractor.git
cd PDFDataExtractor

# 安装依赖
pip install -r requirements.txt

# 如果使用Streamlit界面
pip install -r requirements_streamlit.txt
```

#### 启动Streamlit应用

```bash
streamlit run streamlit_app/streamlit_app.py
```

应用将在浏览器中自动打开（默认地址：http://localhost:8501）

**本地部署支持所有功能，包括Transformer**

### 快速使用示例

#### PDF文件处理

1. 上传PDF文件
2. 选择提取方法（PDFPlumber 或 Camelot）
3. 选择Flavor（auto/lines/text 或 auto/lattice/stream）
4. 点击"开始提取"
5. 查看提取结果

#### 图像文件处理

1. 上传图像文件（PNG、JPG等）
2. 选择检测引擎（PaddleOCR 或 Transformer）
3. 点击"开始提取"
4. 查看提取结果

## 📖 功能说明

### PDF文件处理

- **PDFPlumber**：适合无框表格，支持 lines 和 text 两种模式
- **Camelot**：适合有框表格，支持 lattice 和 stream 两种模式
- **自动参数计算**：根据页面特征自动优化提取参数
- **表格类型识别**：自动判断表格类型并选择最佳方法

### 图像文件处理

- **PaddleOCR**（推荐）：
  - 优秀的中文识别能力
  - 快速的表格检测和结构识别
  - 支持HTML格式输出
  - 适合中文文档处理
  
- **Transformer**：
  - 高精度的表格检测
  - 复杂表格结构识别
  - 适合英文文档和复杂表格

## 📚 文档索引

### 🎯 快速开始

- [快速开始指南](docs/quick_start.md) - 5分钟快速上手
- [部署指南](docs/deployment_guide.md) - 部署和配置说明 ⭐（推荐）
- [Streamlit用户使用指南](docs/streamlit_user_guide.md) - 完整使用说明

### 👥 用户指南

- [常见问题FAQ](docs/FAQ.md) - 常见问题解答
- [部署指南](docs/deployment_guide.md) - 部署和配置说明

### 🔧 模块使用文档

- [提取器使用文档](docs/extractors/usage.md) - Camelot/PDFPlumber提取器
- [引擎使用文档](docs/engines/usage.md) - EasyOCR/Transformer/PaddleOCR引擎
- [PaddleOCR详细文档](docs/engines/paddleocr_usage.md) - PaddleOCR完整指南
- [模块移植指南](docs/porting_guide.md) - 如何移植模块到其他项目

### 📐 技术原理文档

- [参数计算公式](docs/parameter_calculation_formulas.md) - 参数计算原理
- [参数范围说明](docs/parameter_range_documentation.md) - 参数范围设定
- [表格类型分类原理](docs/table_type_classification_principle.md) - 表格类型判断算法
- [Camelot参数计算](docs/camelot_parameter_calculation.md) - Camelot参数详解
- [PDFPlumber参数计算](docs/pdfplumber_parameter_calculation.md) - PDFPlumber参数详解
- [Transformer处理说明](docs/transformer_table_processing.md) - Transformer模型使用

### 📖 提取器指南

- [Camelot提取指南](docs/camelot_table_extraction_guide.md) - Camelot使用指南
- [PDFPlumber提取指南](docs/pdfplumber_table_extraction_guide.md) - PDFPlumber使用指南
- [PDFPlumber文本行分析](docs/pdfplumber_text_lines_analysis.md) - 文本行分析原理

### 🛠️ 开发文档

- [测试指南](docs/testing_guide.md) - 测试方法和示例


## 🏗️ 项目结构

```
PDFDataExtractor/
├── core/                    # 核心模块
│   ├── extractors/          # 表格提取器（Camelot、PDFPlumber）
│   ├── engines/             # OCR/检测引擎（EasyOCR、Transformer、PaddleOCR）
│   ├── processing/          # 处理模块（特征分析、参数计算、类型识别）
│   └── utils/               # 工具模块
├── streamlit_app/           # Streamlit Web界面
│   ├── components/          # UI组件
│   └── streamlit_app.py     # 主应用入口
├── docs/                     # 文档目录
├── tests/                    # 测试文件
└── requirements.txt          # 依赖列表
```

## 🔄 版本历史

查看 [CHANGELOG.md](CHANGELOG.md) 了解详细的版本变更记录。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[添加许可证信息]

## 🔗 相关链接

- [GitHub仓库](https://github.com/livezingy/PDFDataExtractor)
- [问题反馈](https://github.com/livezingy/PDFDataExtractor/issues)
- [技术文档](https://github.com/livezingy/PDFDataExtractor/tree/main/docs)

---

**当前版本**：v2.0.0  
**最后更新**：2025-12-12
