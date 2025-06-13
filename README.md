# PDFDataExtractor

## 项目简介 | Project Overview
PDFDataExtractor 是一个面向 PDF/图片表格自动检测与结构化导出的工具，支持多种检测模型和参数优化，适用于批量、复杂文档的高效处理。

PDFDataExtractor is a tool for automatic table detection and structured export from PDF/images, supporting multiple detection models and parameter optimization, suitable for efficient processing of batch and complex documents.

## 主要特性 | Key Features
- 支持 PDF、图片的表格检测与结构化导出
- 支持 Camelot、Table-Transformer 等多种检测方式
- 可视化检测结果预览
- 参数可配置，支持批量处理
- 现代化 GUI，支持拖拽、批量、参数自定义
- 日志与错误追踪，便于调试和维护

## 快速开始 | Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## 依赖环境 | Requirements
- Python 3.10+
- 主要依赖：Camelot, PyTorch, Transformers, PIL, PySide6, pdfplumber, pytesseract 等

## 模型文件获取与放置说明 | Model Files Notice
本项目依赖较大的模型文件（如 Table-Transformer、Tesseract-OCR），请用户根据下述说明手动下载并放置：

- Table-Transformer: 下载地址见 [官方仓库](https://github.com/microsoft/table-transformer) 或 [HuggingFace](https://huggingface.co/microsoft/table-transformer)。
- Tesseract-OCR: 可从 [Tesseract 官方](https://github.com/tesseract-ocr/tesseract) 或 [各平台发行版](https://github.com/tesseract-ocr/tesseract/wiki) 下载。

请将上述模型文件分别放置于本项目根目录下的 `models/table-transformer/` 和 `models/Tesseract-OCR/` 文件夹内，结构如下：

```
PDFDataExtractor/
├─ main.py
├─ models/
│  ├─ table-transformer/
│  └─ Tesseract-OCR/
```

> **注意**：发布 Release 时请勿上传 models 文件夹，仅在 Release 页面或文档中提供上述下载链接和放置说明。

---

## Project Overview (English)
PDFDataExtractor is a tool for automatic table detection and structured export from PDF/images, supporting multiple detection models and parameter optimization, suitable for efficient processing of batch and complex documents.

## Key Features
- Table detection and structured export for PDF/images
- Supports Camelot, Table-Transformer and more
- Visual preview of detection results
- Configurable parameters, batch processing
- Modern GUI with drag & drop, batch, custom parameters
- Logging and error tracking for debugging and maintenance

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## Requirements
- Python 3.10+
- Main dependencies: Camelot, PyTorch, Transformers, PIL, PySide6, pdfplumber, pytesseract, etc.

## Model Files Notice
This project requires large model files (e.g., Table-Transformer, Tesseract-OCR) which are NOT included in the GitHub repo or Release due to their size. Please download and place them manually as follows:

- Table-Transformer: Download from [official repo](https://github.com/microsoft/table-transformer) or [HuggingFace](https://huggingface.co/microsoft/table-transformer).
- Tesseract-OCR: Download from [Tesseract official](https://github.com/tesseract-ocr/tesseract) or [platform releases](https://github.com/tesseract-ocr/tesseract/wiki).

Place the downloaded files in `models/table-transformer/` and `models/Tesseract-OCR/` under the project root:

```
PDFDataExtractor/
├─ main.py
├─ models/
│  ├─ table-transformer/
│  └─ Tesseract-OCR/
```

> **Note**: Do NOT upload the models folder to GitHub or Release. Only provide download links and placement instructions in the Release page or documentation.

## Typical Usage
- Run the main program, select PDF or image, set parameters, and click "Start Processing"

  ![Processing Flow Screenshot](docs/Images/PDF%20Table%20Extractor_processImage.png)

- Results and preview images are saved in the output directory
- You can preview detection results in the GUI

  ![GUI Preview Screenshot](docs/Images/PDF%20Table%20Extractor_processpdf.png)

## Contribution
Issues and PRs are welcome! For development, please refer to docs/architecture.md and docs/implementation.md.

## License
MIT
