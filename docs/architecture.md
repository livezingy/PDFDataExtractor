# 代码架构说明 | Architecture

## 目录结构 | Directory Structure

- main.py                # 主程序入口 | Main entry
- core/                  # 核心处理模块 | Core processing
  - detection/           # 表格检测 | Table detection
  - models/              # 结构与模型 | Models
  - processing/          # 处理流程 | Processing pipeline
  - utils/               # 工具函数 | Utilities
- gui/                   # 图形界面 | GUI
  - panels/              # 各功能面板 | Panels
- config/                # 配置文件 | Config
- models/                # 预训练模型、OCR等 | Pretrained models
- output/                # 输出目录 | Output
- docs/                  # 文档 | Documentation

## 主要依赖 | Main Dependencies
- PySide6: 现代化 GUI
- Camelot: PDF 表格检测
- Transformers: Table-Transformer 深度学习表格检测
- pdfplumber, pytesseract: OCR 与 PDF 处理
- numpy, PIL, scikit-learn: 数据处理与图像处理

## 模块划分 | Module Overview
- detection: 表格检测算法（传统与深度学习）
- processing: 参数优化、表格结构化、导出
- gui: 文件选择、参数设置、进度与预览
- utils: 日志、路径、文件等通用工具

## 依赖关系 | Dependency Graph
- main.py 调用 gui 和 core
- gui 通过 core/processing 触发检测与导出
- core/processing 依赖 detection、models、utils
- output/ 由 core/processing 统一管理

---

# Architecture (English)

## Directory Structure
- main.py                # Main entry
- core/                  # Core processing modules
  - detection/           # Table detection
  - models/              # Models and structures
  - processing/          # Processing pipeline
  - utils/               # Utilities
- gui/                   # GUI
  - panels/              # Panels
- config/                # Config files
- models/                # Pretrained models, OCR
- docs/                  # Documentation

## Main Dependencies
- PySide6: Modern GUI
- Camelot: PDF table detection
- Transformers: Table-Transformer deep learning detection
- pdfplumber, pytesseract: OCR and PDF processing
- numpy, PIL, scikit-learn: Data/image processing

## Module Overview
- detection: Table detection (traditional & DL)
- processing: Parameter optimization, structuring, export
- gui: File selection, parameter config, progress, preview
- utils: Logging, path, file utilities

## Dependency Graph
- main.py calls gui and core
- gui triggers detection/export via core/processing
- core/processing depends on detection, models, utils
- output/ managed by core/processing
