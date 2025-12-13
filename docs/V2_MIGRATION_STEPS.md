# V2.0 代码迁移和推送步骤

## 📍 当前状态

- **V2.0 代码位置**：`D:\UP_UP_WORK\2025\PDFDataExtractor`（非 Git 仓库）
- **Git 仓库位置**：`d:\UP_UP_WORK\Github\PDFDataExtractor`（GitHub 远程仓库）
- **V2.0 代码状态**：
  - ✅ 已集成 PaddleOCR
  - ✅ 已移除 GUI 目录
  - ✅ 已模块化重构（engines/extractors）
  - ✅ CHANGELOG.md 已更新（2025-12-12）

## 🎯 操作目标

将 V2.0 代码从 `D:\UP_UP_WORK\2025\PDFDataExtractor` 迁移到 Git 仓库并推送到 GitHub。

## 📋 详细操作步骤

### 步骤 1：备份当前 Git 仓库（推荐）

在开始之前，创建一个备份分支以防万一：

```powershell
cd d:\UP_UP_WORK\Github\PDFDataExtractor

# 创建备份分支
git branch backup-before-v2.0

# 查看当前分支状态
git status
```

### 步骤 2：对比两个目录的关键差异

**V2.0 代码的新增/变更：**
- ✅ 新增 `core/engines/` 目录（包含 PaddleOCR 引擎）
- ✅ 新增 `core/extractors/` 目录（模块化提取器）
- ✅ 移除 `gui/` 目录
- ✅ 更新 `requirements_streamlit.txt`（添加 PaddleOCR）
- ✅ 更新 `CHANGELOG.md`（日期：2025-12-12）
- ✅ 新增多个文档文件

**当前 Git 仓库需要保留：**
- `.git/` 目录（Git 历史）
- `.gitignore` 文件
- 可能有一些 V2.0 没有的配置文件

### 步骤 3：复制 V2.0 代码到 Git 仓库

**方法 A：使用 PowerShell 复制（推荐）**

```powershell
# 进入 Git 仓库目录
cd d:\UP_UP_WORK\Github\PDFDataExtractor

# 备份 .git 目录（临时）
Copy-Item -Path .\.git -Destination .\.git.backup -Recurse

# 复制 V2.0 代码（排除 .git 目录）
$exclude = @('.git', '.git.backup')
Get-ChildItem -Path "D:\UP_UP_WORK\2025\PDFDataExtractor" -Recurse | 
    Where-Object { $_.FullName -notmatch '\.git' } | 
    ForEach-Object {
        $destPath = $_.FullName.Replace("D:\UP_UP_WORK\2025\PDFDataExtractor", "d:\UP_UP_WORK\Github\PDFDataExtractor")
        $destDir = Split-Path -Parent $destPath
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Copy-Item -Path $_.FullName -Destination $destPath -Force
    }

# 恢复 .git 目录
Remove-Item -Path .\.git -Recurse -Force
Move-Item -Path .\.git.backup -Destination .\.git
```

**方法 B：手动复制（更安全，推荐用于首次操作）**

1. 打开文件管理器
2. 复制 `D:\UP_UP_WORK\2025\PDFDataExtractor` 中的所有文件和文件夹
3. 粘贴到 `d:\UP_UP_WORK\Github\PDFDataExtractor`
4. **重要**：如果提示覆盖，选择"跳过" `.git` 目录
5. 或者先删除 Git 仓库中不需要的文件，再复制

**方法 C：使用 robocopy（Windows 内置，更可靠）**

```powershell
cd d:\UP_UP_WORK\Github\PDFDataExtractor

# 备份 .git
Copy-Item -Path .\.git -Destination .\.git.backup -Recurse

# 使用 robocopy 复制（排除 .git）
robocopy "D:\UP_UP_WORK\2025\PDFDataExtractor" "d:\UP_UP_WORK\Github\PDFDataExtractor" /E /XD .git .git.backup /XF .gitignore

# 恢复 .git
Remove-Item -Path .\.git -Recurse -Force
Move-Item -Path .\.git.backup -Destination .\.git

# 确保 .gitignore 存在（如果需要）
Copy-Item "D:\UP_UP_WORK\2025\PDFDataExtractor\.gitignore" "d:\UP_UP_WORK\Github\PDFDataExtractor\.gitignore" -Force
```

### 步骤 4：验证关键文件

复制完成后，验证以下关键文件：

```powershell
cd d:\UP_UP_WORK\Github\PDFDataExtractor

# 1. 检查 GUI 目录是否已删除
Test-Path .\gui
# 应该返回 False

# 2. 检查 PaddleOCR 引擎是否存在
Test-Path .\core\engines\paddleocr_engine.py
# 应该返回 True

# 3. 检查 requirements_streamlit.txt 是否包含 PaddleOCR
Select-String -Path .\requirements_streamlit.txt -Pattern "paddleocr"
# 应该找到匹配

# 4. 检查 CHANGELOG.md 日期
Select-String -Path .\docs\CHANGELOG.md -Pattern "2025-12-12"
# 应该找到匹配

# 5. 检查 Git 状态
git status
```

### 步骤 5：处理 .gitignore 文件

确保 `.gitignore` 文件正确：

```powershell
# 查看 .gitignore 内容
Get-Content .\.gitignore

# 如果 V2.0 的 .gitignore 更完整，可以合并或替换
```

### 步骤 6：查看更改内容

```powershell
cd d:\UP_UP_WORK\Github\PDFDataExtractor

# 查看所有更改
git status

# 查看新增的文件
git status --short | Select-String "^??"

# 查看修改的文件
git status --short | Select-String "^ M"
```

### 步骤 7：添加所有更改

```powershell
# 添加所有新文件和修改的文件
git add .

# 再次查看状态
git status
```

### 步骤 8：提交更改

```powershell
git commit -m "feat: 发布 v2.0.0 - 集成 PaddleOCR，移除 GUI 界面

主要变更：
- ✨ 集成 PaddleOCR (PP-Structure) 表格检测和 OCR 引擎
- ✨ 模块化架构：分离 engines 和 extractors
- ✨ 移除 PySide6 GUI 界面，仅保留 Streamlit
- ✨ 更新依赖文件，添加 PaddleOCR 支持
- ✨ 优化 Streamlit 界面和用户体验
- 📚 更新文档和 CHANGELOG

详细变更请参考 CHANGELOG.md"
```

### 步骤 9：推送到 GitHub

```powershell
# 推送到 main 分支
git push origin main

# 如果遇到冲突，先拉取最新代码
# git pull origin main
# 解决冲突后再次推送
```

### 步骤 10：创建版本标签

```powershell
# 创建 v2.0.0 标签
git tag -a v2.0.0 -m "版本 2.0.0 - 集成 PaddleOCR，移除 GUI 界面

主要特性：
- PaddleOCR (PP-Structure) 集成
- 模块化架构重构
- Streamlit 界面优化
- 移除 GUI 界面"

# 推送标签到 GitHub
git push origin v2.0.0

# 或者推送所有标签
git push origin --tags
```

### 步骤 11：创建 GitHub Release（可选但推荐）

1. 访问：https://github.com/livezingy/PDFDataExtractor/releases/new
2. 选择标签：`v2.0.0`
3. 标题：`v2.0.0 - PaddleOCR 集成版本`
4. 描述：从 `docs/CHANGELOG.md` 复制 v2.0.0 部分的内容
5. 点击"发布 Release"

## ⚠️ 注意事项

### 1. 保留 Git 历史
- **重要**：复制文件时不要覆盖 `.git` 目录
- 如果意外覆盖，可以从备份恢复

### 2. 处理冲突文件
如果某些文件在两个目录中都存在但内容不同：
- `.gitignore`：建议使用 V2.0 的版本，但检查是否需要保留 Git 仓库特有的规则
- `README.md`：建议使用 V2.0 的版本
- `CHANGELOG.md`：建议使用 V2.0 的版本（已更新日期）

### 3. 检查敏感信息
确保没有提交敏感信息：
- API 密钥
- 个人配置
- 临时文件

### 4. 测试验证
推送前建议：
- 检查关键文件是否存在
- 验证 PaddleOCR 相关代码
- 确认 GUI 目录已删除

## 🔍 验证清单

推送完成后，验证以下内容：

- [ ] GitHub 仓库显示最新提交
- [ ] 标签 `v2.0.0` 已创建
- [ ] `gui/` 目录在 GitHub 上已删除
- [ ] `core/engines/paddleocr_engine.py` 已存在
- [ ] `requirements_streamlit.txt` 包含 PaddleOCR
- [ ] `CHANGELOG.md` 日期为 2025-12-12
- [ ] Release 已创建（如果执行了步骤 11）

## 🐛 常见问题处理

### Q1: 复制时提示文件被占用
**解决**：关闭可能打开这些文件的程序（IDE、编辑器等）

### Q2: Git 状态显示大量删除
**解决**：这是正常的，因为移除了 GUI 目录。确认后继续提交。

### Q3: 推送时提示需要先拉取
**解决**：
```powershell
git pull origin main --rebase
# 解决冲突后
git push origin main
```

### Q4: 想回退到之前版本
**解决**：
```powershell
# 切换到备份分支
git checkout backup-before-v2.0

# 或者查看提交历史
git log --oneline
```

## 📝 快速命令清单

```powershell
# 完整流程（如果使用 robocopy 方法）
cd d:\UP_UP_WORK\Github\PDFDataExtractor
git branch backup-before-v2.0
Copy-Item -Path .\.git -Destination .\.git.backup -Recurse
robocopy "D:\UP_UP_WORK\2025\PDFDataExtractor" "d:\UP_UP_WORK\Github\PDFDataExtractor" /E /XD .git .git.backup
Remove-Item -Path .\.git -Recurse -Force
Move-Item -Path .\.git.backup -Destination .\.git
git add .
git commit -m "feat: 发布 v2.0.0 - 集成 PaddleOCR"
git push origin main
git tag -a v2.0.0 -m "版本 2.0.0"
git push origin v2.0.0
```

---

**最后更新**：2025-01-XX  
**维护者**：项目团队
