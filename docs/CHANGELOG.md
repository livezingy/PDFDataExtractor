# 更新日志

所有重要的项目变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [2.0.0] - 2025-12-XX (计划中)

### 重大变更
- **移除GUI界面**：删除PySide6 GUI，仅保留Streamlit界面
- **集成PaddleOCR**：添加PP-Structure表格检测和OCR功能

### 新增
- PaddleOCR (PP-Structure) 表格检测器
- PaddleOCR表格识别和OCR引擎
- 布局分析功能（基础）
- 多文件批量上传（Streamlit）
- 增强的进度显示

### 改进
- Streamlit界面优化
- 表格检测准确率提升
- OCR识别准确率提升（特别是中文和混合语言）

### 移除
- GUI界面 (PySide6)
- PySide6相关依赖

### 变更
- 默认OCR引擎：PaddleOCR（Tesseract/EasyOCR作为备选）

## [1.5.0] - 2025-XX-XX

### 新增
- GUI界面支持
- Streamlit界面支持
- Camelot和PDFPlumber表格提取
- Tesseract和EasyOCR支持

### 改进
- 参数自动计算
- 表格类型分类
- 结果可视化

## 版本历史

- **v2.0+**：仅Streamlit，集成PaddleOCR
- **v1.x**：GUI + Streamlit，使用Camelot/PDFPlumber

详细版本信息请参考 [Git Tags](../../tags)