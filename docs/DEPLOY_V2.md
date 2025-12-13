# V2.0 代码推送指南

本文档说明如何将已完成的 V2.0 代码（集成 PaddleOCR）推送到 GitHub。

## 📋 前置条件检查

在开始之前，请确认：

- ✅ V2.0 代码已完成开发并测试通过
- ✅ 代码已集成 PaddleOCR (PP-Structure)
- ✅ GUI 界面已移除（仅保留 Streamlit）
- ✅ 所有功能已测试完成

## 🚀 推送步骤

### 步骤 1：确认 V2.0 代码位置

**情况 A：代码在其他目录**
```powershell
# 如果 V2.0 代码在其他位置，需要先复制到当前工作目录
# 例如：从 D:\Development\PDFDataExtractor_V2 复制到当前目录
```

**情况 B：代码在其他分支**
```powershell
cd d:\UP_UP_WORK\Github\PDFDataExtractor
git branch -a  # 查看所有分支
git checkout <v2.0分支名>  # 如果存在 v2.0 分支
```

**情况 C：代码在当前目录但未提交**
```powershell
cd d:\UP_UP_WORK\Github\PDFDataExtractor
git status  # 查看当前状态
```

### 步骤 2：更新代码到当前文件夹

如果 V2.0 代码在其他位置，需要：

1. **备份当前代码**（可选，但推荐）
   ```powershell
   git branch backup-before-v2.0
   ```

2. **复制 V2.0 代码**
   - 将 V2.0 代码的所有文件复制到当前目录
   - 覆盖现有文件（保留 `.git` 文件夹）

3. **验证关键文件**
   - 检查 `requirements_streamlit.txt` 是否包含 `paddlepaddle` 和 `paddleocr`
   - 检查 `gui/` 目录是否已删除
   - 检查代码中是否使用了 PaddleOCR

### 步骤 3：更新依赖文件

确保 `requirements_streamlit.txt` 包含 PaddleOCR 依赖：

```txt
# PaddleOCR 相关依赖
paddlepaddle>=2.5.0
paddleocr>=2.7.0
```

如果使用 CPU 版本：
```txt
paddlepaddle-cpu>=2.5.0
paddleocr>=2.7.0
```

### 步骤 4：检查并清理不需要的文件

根据 CHANGELOG，V2.0 应该：

- ✅ **移除**：`gui/` 目录及其所有文件
- ✅ **移除**：`requirements.txt` 中的 PySide6 依赖（或删除整个文件）
- ✅ **保留**：`streamlit_app/` 目录
- ✅ **更新**：`requirements_streamlit.txt` 添加 PaddleOCR

检查命令：
```powershell
# 检查 GUI 目录是否存在
Test-Path .\gui

# 检查 requirements.txt 是否包含 PySide6
Select-String -Path .\requirements.txt -Pattern "PySide6"
```

### 步骤 5：更新 CHANGELOG.md

更新 CHANGELOG.md 中的日期：

```markdown
## [2.0.0] - 2025-01-XX  # 将 XX 改为实际日期
```

### 步骤 6：提交更改

```powershell
cd d:\UP_UP_WORK\Github\PDFDataExtractor

# 1. 查看所有更改
git status

# 2. 添加所有更改的文件
git add .

# 3. 提交更改（使用清晰的提交信息）
git commit -m "feat: 发布 v2.0.0 - 集成 PaddleOCR，移除 GUI 界面

- 集成 PaddleOCR (PP-Structure) 表格检测和 OCR
- 移除 PySide6 GUI 界面，仅保留 Streamlit
- 更新依赖文件，添加 PaddleOCR 支持
- 优化 Streamlit 界面和用户体验
- 提升表格检测和 OCR 识别准确率"
```

### 步骤 7：推送到 GitHub

```powershell
# 推送到 main 分支
git push origin main

# 如果遇到冲突，先拉取最新代码
git pull origin main
# 解决冲突后再次推送
git push origin main
```

### 步骤 8：创建版本标签

```powershell
# 创建 v2.0.0 标签
git tag -a v2.0.0 -m "版本 2.0.0 - 集成 PaddleOCR，移除 GUI"

# 推送标签到 GitHub
git push origin v2.0.0

# 或者推送所有标签
git push origin --tags
```

### 步骤 9：创建 Release（可选）

在 GitHub 网页上：

1. 访问：https://github.com/livezingy/PDFDataExtractor/releases/new
2. 选择标签：`v2.0.0`
3. 标题：`v2.0.0 - PaddleOCR 集成版本`
4. 描述：从 CHANGELOG.md 复制 v2.0.0 的内容
5. 发布 Release

## 🔍 验证步骤

推送完成后，验证：

1. **检查 GitHub 仓库**
   - 访问：https://github.com/livezingy/PDFDataExtractor
   - 确认最新提交已推送
   - 确认标签 `v2.0.0` 已创建

2. **检查代码内容**
   - 确认 `gui/` 目录已删除
   - 确认 `requirements_streamlit.txt` 包含 PaddleOCR
   - 确认代码中使用了 PaddleOCR

3. **检查文档**
   - 确认 `CHANGELOG.md` 日期已更新
   - 确认 `README.md` 已更新（如果需要）

## ⚠️ 注意事项

1. **备份重要数据**：推送前建议创建备份分支
2. **测试验证**：确保所有功能在推送前已测试
3. **依赖兼容性**：确保 PaddleOCR 依赖与 Streamlit Cloud 兼容
4. **文档同步**：确保所有文档（README、CHANGELOG）已更新

## 🐛 常见问题

### Q: 如果推送时出现冲突怎么办？

A: 
```powershell
# 拉取最新代码
git pull origin main

# 解决冲突后
git add .
git commit -m "解决合并冲突"
git push origin main
```

### Q: 如何回退到之前的版本？

A:
```powershell
# 查看提交历史
git log --oneline

# 回退到指定提交
git reset --hard <commit-hash>

# 强制推送（谨慎使用）
git push origin main --force
```

### Q: 如何只推送部分文件？

A:
```powershell
# 添加特定文件
git add <文件路径>

# 提交并推送
git commit -m "更新特定文件"
git push origin main
```

## 📝 快速命令清单

```powershell
# 完整推送流程（如果代码已在当前目录）
cd d:\UP_UP_WORK\Github\PDFDataExtractor
git status
git add .
git commit -m "feat: 发布 v2.0.0 - 集成 PaddleOCR"
git push origin main
git tag -a v2.0.0 -m "版本 2.0.0"
git push origin v2.0.0
```

---

**最后更新**：2025-01-XX
**维护者**：项目团队
