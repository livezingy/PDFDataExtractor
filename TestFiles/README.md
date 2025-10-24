# PDF表格提取测试数据集

## 目录结构

```
TestFiles/
├── bordered/          # 有边框表格
│   ├── edge_tol.pdf           # 边缘容差测试
│   ├── financial_report.pdf   # 财务报告表格
│   ├── sample_bordered.pdf    # 标准有边框表格
│   └── short_lines.pdf        # 短线条表格
├── unbordered/        # 无边框表格
│   ├── foo.pdf                # 通用测试文件
│   ├── sample_unbordered.pdf  # 标准无边框表格
│   └── scientific_data.pdf    # 科学数据表格
├── complex/           # 复杂表格
│   ├── sample_complex.pdf     # 合并单元格表格
│   └── superscript.pdf        # 上标格式表格
└── scanned/           # 扫描文档（待补充）
```

## 测试文件说明

### 有边框表格 (bordered/)
- **edge_tol.pdf**: 测试边缘容差参数的有边框表格
- **financial_report.pdf**: 财务报告格式，包含数字和百分比
- **sample_bordered.pdf**: 标准的有边框表格，包含员工信息
- **short_lines.pdf**: 短线条表格，测试线条检测阈值

### 无边框表格 (unbordered/)
- **foo.pdf**: 通用测试文件，可能包含无边框表格
- **sample_unbordered.pdf**: 标准无边框表格，产品信息
- **scientific_data.pdf**: 科学实验数据，包含数值和p值

### 复杂表格 (complex/)
- **sample_complex.pdf**: 包含合并单元格的复杂表格
- **superscript.pdf**: 包含上标等特殊格式的表格

### 扫描文档 (scanned/)
- 待补充：需要添加扫描PDF或图像文件

## 测试目标

1. **参数验证**: 验证自动参数计算是否优于默认参数
2. **方法对比**: 测试Camelot vs PDFPlumber在不同表格类型上的表现
3. **Transformer验证**: 测试深度学习+OCR管线的基本功能
4. **评估体系**: 使用TableEvaluator进行量化评分

## 使用说明

每个测试文件都应该通过以下方法进行测试：
- Camelot lattice (有边框表格)
- Camelot stream (无边框表格)
- PDFPlumber lines (有边框表格)
- PDFPlumber text (无边框表格)
- Transformer + OCR (所有类型，特别是扫描文档)

测试结果将记录在 `tests/results/` 目录中。
