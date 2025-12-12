# v1.x 到 v2.0 迁移指南

## 概述

v2.0版本移除了GUI界面，仅保留Streamlit界面，并集成了PaddleOCR。

## 主要变更

### 界面变更
- ❌ 移除：GUI界面 (PySide6)
- ✅ 保留：Streamlit界面（增强版）

### 功能变更
- ✅ 新增：PaddleOCR (PP-Structure) 表格检测和OCR
- ✅ 保留：Camelot和PDFPlumber表格提取
- ✅ 保留：Tesseract和EasyOCR（作为备选）

### 依赖变更
- ❌ 移除：PySide6
- ✅ 新增：paddlepaddle, paddleocr

## 迁移步骤

### 1. 备份当前配置

```bash
# 备份配置文件
cp config/config.json config/config.json.backup
```

### 2. 更新代码

```bash
# 拉取v2.0代码
git checkout main
git pull origin main
```

### 3. 更新依赖

```bash
# 卸载旧依赖
pip uninstall PySide6

# 安装新依赖
pip install -r requirements.txt
```

### 4. 更新配置

- 检查配置文件是否需要更新
- 参考新的配置示例

### 5. 测试

- 测试Streamlit界面
- 验证表格提取功能
- 测试PaddleOCR功能

## 常见问题

### Q: 如何继续使用GUI版本？
A: 切换到v1.x维护分支或archive分支：
```bash
git checkout v1.x-maintenance
# 或
git checkout archive/gui-version
```

### Q: v1.x版本还会更新吗？
A: 仅进行bug修复，不添加新功能。维护期限：2026-06-30

### Q: 如何报告v1.x的bug？
A: 在GitHub Issues中标注版本为v1.x

## 获取帮助

- 查看 [文档](../docs/)
- 提交 [Issue](../../issues)
- 查看 [FAQ](../docs/FAQ.md)