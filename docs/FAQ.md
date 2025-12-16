# 常见问题FAQ

## 部署和使用

### Q0: 如何快速试用，无需安装？

**A**: 使用Streamlit Cloud一键部署：

1. 访问 [Streamlit Cloud](https://streamlit.io/cloud)
2. 连接GitHub账户
3. 选择仓库：`livezingy/PDFDataExtractor`
4. 点击"Deploy"

详细步骤请参考 [Streamlit Cloud部署指南](streamlit_cloud_deployment.md)

**可用功能（Streamlit Cloud）**：
- ✅ PDFPlumber（PDF表格提取）
- ✅ Camelot（PDF表格提取）
- ❌ **PaddleOCR+PP-Structure**：需要在本地/服务器部署（模型过大，Streamlit Cloud资源限制）
- ❌ Transformer：需要在本地/服务器部署（资源限制）

**完整功能（本地/服务器部署）**：
- ✅ PDFPlumber、Camelot
- ✅ PaddleOCR+PP-Structure（图像表格检测和结构识别）
- ✅ Transformer（图像表格检测和结构识别）

## 安装问题

### Q1: 如何安装项目依赖？

**A**: 使用以下命令安装：

```bash
# 安装核心依赖
pip install -r requirements.txt

# 安装Streamlit界面依赖
pip install -r requirements_streamlit.txt
```

**注意**：如果遇到依赖冲突，建议使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

### Q2: 安装PaddleOCR失败怎么办？

**A**: PaddleOCR需要PaddlePaddle框架，安装步骤：

```bash
# 先安装PaddlePaddle（CPU版本）
pip install paddlepaddle

# 再安装PaddleOCR
pip install paddleocr

# 如果使用GPU
pip install paddlepaddle-gpu
```

**常见问题**：
- 如果提示找不到paddlepaddle，确保Python版本>=3.8
- 如果内存不足，尝试使用CPU版本

### Q3: Camelot安装失败？

**A**: Camelot依赖Ghostscript，需要先安装系统依赖：

**Windows**：
- 下载并安装 [Ghostscript](https://www.ghostscript.com/download/gsdnld.html)
- 确保Ghostscript在系统PATH中

**Linux**：
```bash
sudo apt-get install ghostscript
```

**Mac**：
```bash
brew install ghostscript
```

### Q4: Transformer模型在哪里下载？

**A**: Transformer模型会在首次使用时自动下载，但可能较慢。

**手动下载**（可选）：
- Detection模型：`microsoft/table-transformer-detection`
- Structure模型：`microsoft/table-transformer-structure-recognition`

使用Hugging Face CLI：
```bash
pip install huggingface_hub
huggingface-cli download microsoft/table-transformer-detection
huggingface-cli download microsoft/table-transformer-structure-recognition
```

## 使用问题

### Q5: 如何选择最佳的提取方法？

**A**: 选择建议：

1. **有框表格** → Camelot (lattice)
2. **无框表格** → PDFPlumber (text)
3. **不确定** → 使用 auto 模式，让系统自动选择

**判断方法**：
- 查看PDF，如果表格有明显的边框线，使用Camelot
- 如果表格没有边框，只有文本对齐，使用PDFPlumber
- 参考 [表格类型判断指南](table_type_judgment_guide.md)

### Q6: 图像文件应该选择哪个引擎？

**A**: 选择建议：

- **Streamlit Cloud环境** → ❌ **图像表格检测功能不可用**（需要在本地/服务器部署）
- **本地/服务器部署**：
  - **中文文档** → PaddleOCR+PP-Structure（推荐）
  - **英文文档/复杂表格** → Transformer 或 PaddleOCR+PP-Structure
  - **需要快速处理** → PaddleOCR+PP-Structure

**对比**：
| 特性 | PaddleOCR+PP-Structure | Transformer |
|------|------------------------|-------------|
| 中文识别 | ✅ 优秀 | ❌ 不支持 |
| 处理速度 | ✅ 快 | ❌ 慢 |
| 复杂表格 | ✅ 良好 | ✅ 优秀 |
| 模型大小 | ❌ 较大（200-500MB+） | ❌ 较大 |
| Streamlit Cloud | ❌ **不可用**（需本地/服务器） | ❌ 不可用（需本地/服务器） |
| 本地/服务器部署 | ✅ 推荐 | ✅ 推荐 |

**重要说明**：
- PaddleOCR+PP-Structure 需要下载多个大模型（PP-StructureV3、PaddleX等），在Streamlit Cloud环境下会频繁超时
- **图像表格检测功能必须在本地或服务器部署才能使用**

### Q7: 提取结果为空怎么办？

**A**: 排查步骤：

1. **检查文件**：
   - 确认PDF/图像中确实有表格
   - 检查文件是否损坏
   - 尝试其他文件

2. **调整方法**：
   - 切换提取方法（PDFPlumber ↔ Camelot）
   - 尝试不同的Flavor选项
   - 使用 auto 模式

3. **检查参数**：
   - 如果使用手动模式，尝试调整参数
   - 使用自动参数模式（推荐）

4. **文件质量**：
   - 扫描的PDF可能需要OCR预处理
   - 图像质量差的文件可能无法识别

### Q8: 处理速度很慢怎么办？

**A**: 优化建议：

1. **选择更快的引擎**：
   - PDF文件：优先使用PDFPlumber
   - 图像文件：优先使用PaddleOCR

2. **文件大小**：
   - 处理较小的文件
   - 分页处理大文件

3. **硬件加速**：
   - 如果可用，启用GPU加速
   - 使用MKLDNN优化（Intel CPU）

4. **首次使用**：
   - 首次使用需要下载模型，这是正常的
   - 后续使用会更快

### Q9: 如何提高提取准确率？

**A**: 优化建议：

1. **选择正确的方法**：
   - 根据表格类型选择方法
   - 参考 [表格类型判断指南](table_type_judgment_guide.md)

2. **参数调整**：
   - 使用自动参数模式（推荐）
   - 如果结果不理想，尝试手动调整

3. **文件质量**：
   - 使用高质量的PDF/图像
   - 避免模糊、倾斜的文档

4. **预处理**：
   - 对于扫描文档，可能需要OCR预处理
   - 确保图像清晰

## 错误处理

### Q10: "ModuleNotFoundError: No module named 'xxx'"

**A**: 这是依赖缺失问题：

```bash
# 重新安装依赖
pip install -r requirements.txt

# 如果特定模块缺失，单独安装
pip install xxx
```

### Q11: "PaddleOCR table detection failed: PPStructure.call() got an unexpected keyword argument 'layout'"

**A**: 这是已知问题，已在v2.0.0修复。

**解决方案**：
- 更新到最新版本
- 参考 [Bug修复记录](bugfixes/paddleocr_layout_parameter_fix.md)

### Q12: "PageFeatureAnalyzer object has no attribute '_get_mode_with_fallback'"

**A**: 这是静态方法调用问题，已在v2.0.0修复。

**解决方案**：
- 更新到最新版本
- 确保使用正确的类方法调用

### Q13: Streamlit启动失败

**A**: 检查步骤：

1. **检查Streamlit安装**：
```bash
pip install streamlit
```

2. **检查端口占用**：
```bash
# 如果8501端口被占用，使用其他端口
streamlit run streamlit_app/streamlit_app.py --server.port 8502
```

3. **检查Python版本**：
```bash
python --version  # 应该是3.8+
```

### Q14: 文件上传后无法处理

**A**: 检查项：

1. **文件大小**：确保不超过10MB（测试版本限制）
2. **文件格式**：支持PDF和图像格式（PNG、JPG等）
3. **文件完整性**：确保文件未损坏
4. **权限问题**：确保有读取文件的权限

## 性能问题

### Q15: 内存占用过高

**A**: 优化建议：

1. **处理较小文件**：一次处理一页或较小的文件
2. **关闭不需要的引擎**：只加载使用的引擎
3. **使用CPU版本**：如果不需要GPU，使用CPU版本
4. **清理缓存**：定期清理模型缓存

### Q16: GPU未使用

**A**: 检查步骤：

1. **检查GPU可用性**：
```python
import torch
print(torch.cuda.is_available())
```

2. **配置GPU使用**：
```python
# 对于PaddleOCR
engine = EngineFactory.create_detection('paddleocr', use_gpu=True)

# 对于Transformer
# 确保CUDA正确安装
```

3. **检查驱动**：确保NVIDIA驱动和CUDA正确安装

## 其他问题

### Q17: 如何贡献代码？

**A**: 贡献步骤：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

详细说明请查看GitHub仓库的贡献指南。

### Q18: 如何报告Bug？

**A**: 报告方式：

1. 在 [GitHub Issues](https://github.com/livezingy/PDFDataExtractor/issues) 创建Issue
2. 提供详细信息：
   - 错误信息
   - 复现步骤
   - 环境信息（Python版本、操作系统等）
   - 相关文件（如果可能）

### Q19: 如何获取技术支持？

**A**: 获取帮助：

1. 查看本文档（FAQ）
2. 查看 [用户使用指南](streamlit_user_guide.md)
3. 查看 [技术文档](../README.md#-文档索引)
4. 提交 [GitHub Issue](https://github.com/livezingy/PDFDataExtractor/issues)

### Q20: 项目支持哪些语言？

**A**: 支持情况：

- **界面语言**：目前主要是中文和英文
- **OCR语言**：
  - PaddleOCR：支持中文、英文等多种语言
  - EasyOCR：支持80+种语言
  - Transformer：主要用于英文

---

**提示**：如果问题未在此列出，请查看 [用户使用指南](streamlit_user_guide.md) 或提交 [GitHub Issue](https://github.com/livezingy/PDFDataExtractor/issues)。
