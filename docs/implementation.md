# 实现逻辑说明 | Implementation

## 主要功能实现逻辑
- 支持 PDF/图片的表格检测与结构化导出，自动选择最优检测方式
- 并行调用 Camelot 和 Table-Transformer 检测表格区域
- 匹配检测结果，优先采用高置信度 Camelot 结果，自动参数优化
- 支持参数自定义、批量处理、可视化预览
- 结果导出为 CSV/JSON，输出结构清晰

## 关键算法流程
### 表格检测与提取流程（Table Detection and Extraction Workflow）
对于每个 PDF 文件和每个选定页面，系统执行如下步骤：

1. **并行表格检测**
   - 使用 Camelot 按所选模式（lattice、stream、hybrid、network）检测表格
   - 使用 Table-Transformer（深度学习模型）检测页面图像中的表格区域
   - 两种检测方法并行运行，提高效率

2. **表格区域匹配与决策逻辑**
   - 对于每个 transformer 检测到的表格区域，系统尝试找到最佳匹配的 Camelot 表格（坐标转换后）
   - 如果找到匹配且 Camelot 表格准确率高于可配置阈值：
     - 直接采用 Camelot 结果
   - 如果 Camelot 表格准确率低于阈值：
     - 系统自动优化 Camelot 参数重新提取
     - 若仍不理想，则回退为 transformer 区域+OCR
   - 若未找到匹配 Camelot 表格：
     - 直接用 transformer 区域+OCR 提取表格内容

3. **去重与补充**
   - 合并去重所有检测结果，补充高准确度未匹配的 Camelot 表格

4. **参数优化**
   - 若 Camelot 初始提取不理想，自动分析表格结构并动态调整参数，最多迭代若干次

5. **结果导出与可视化**
   - 导出结构化表格（CSV/JSON）
   - 保存检测可视化图片，所有输出按 PDF 分文件夹管理

### 使用 TableTransformerForObjectDetection 从图片中提取表格数据

本节演示如何使用基于 Transformer 的表格结构识别（TSR）模型 `TableTransformerForObjectDetection`，从图片中提取结构化表格数据。
- [TableTransformer 示例代码](https://huggingface.co/spaces/SalML/TableTransformer2CSV/blob/main/app.py)

内容包括：
1. 结构识别模型输出的含义
2. 从检测结果重建表格数据的详细步骤

#### 1. TableTransformerForObjectDetection 输出说明

将图片输入 TableTransformerForObjectDetection 结构识别模型后，经过后处理，通常会得到如下输出：

- **probas**：形状为 [N, num_classes] 的张量，N 为检测到的对象数（如表格行、列等），每行为各类别的 softmax 概率
- **bboxes_scaled**：形状为 [N, 4] 的张量，每行为 [xmin, ymin, xmax, ymax]，表示检测对象的边界框坐标，已映射到原图尺寸
- **model.config.id2label**：类别索引到可读标签（如 'table row', 'table column'）的映射

对于每个检测对象（行/列）：
- `probas` 中概率最大的类别即为该对象类型（行、列等）
- `bboxes_scaled` 中对应的坐标为该对象在图片中的位置

#### 2. 检测结果重建表格的详细步骤

以下步骤描述如何根据模型输出重建表格结构并提取单元格数据：

##### 步骤1：分离行和列
- 根据类别标签将边界框分为行和列
- 按垂直（y）坐标排序行，按水平（x）坐标排序列

##### 步骤2：裁剪每一行及单元格
- 对每个行边界框，从表格图片中裁剪出该行区域
- 对每一行，利用列边界框进一步裁剪出每个单元格
- 行与列边界框的交集即为单元格区域

##### 步骤3：对每个单元格进行 OCR
- 使用 OCR 引擎（如 Tesseract）识别单元格图片中的文本

##### 步骤4：组装表格
- 按行列顺序将识别结果组合为二维数组或 pandas DataFrame

## 参数传递与数据流
- 用户参数由 GUI 收集，贯穿 pipeline
- 所有输出、日志、预览均按 output_path 组织
- 日志与异常处理贯穿全流程

---

# Implementation (English)

## Main Logic
- Supports table detection/structuring for PDF/images, auto-selects best method
- Runs Camelot and Table-Transformer in parallel for table region detection
- Matches results, prefers high-confidence Camelot, auto-optimizes parameters
- Supports custom parameters, batch processing, visual preview
- Exports results as CSV/JSON, clear output structure

## Key Algorithm Flow
### Table Detection and Extraction Workflow
For each PDF file and each selected page, the system performs the following steps:

1. **Parallel Table Detection**
   - Camelot is used to detect tables using the selected flavor (lattice, stream, hybrid, or network)
   - Table-Transformer (deep learning model) is used to detect table regions in the page image
   - Both detection methods run in parallel for efficiency

2. **Table Region Matching and Decision Logic**
   - For each table region detected by the transformer, the system attempts to find the best-matching Camelot table (after coordinate transformation)
   - If a matching Camelot table is found and its accuracy is above a configurable threshold:
     - The Camelot result is used directly
   - If the Camelot table's accuracy is below the threshold:
     - The system attempts to re-extract the table using dynamically optimized Camelot parameters
     - If the result is still unsatisfactory, the system falls back to using the transformer region with OCR-based extraction
   - If no matching Camelot table is found:
     - The transformer region is processed with OCR to extract the table content
   - [Example code for TableTransformer structure extraction](https://huggingface.co/spaces/SalML/TableTransformer2CSV/blob/main/app.py)

3. **Deduplication and Supplementation**
   - Results are deduplicated to ensure each table is output only once
   - High-accuracy Camelot tables not matched to any transformer region are also included in the final results

4. **Parameter Optimization**
   - If Camelot's initial extraction is not accurate enough, the system analyzes the table's structure and dynamically adjusts parameters
   - This process is repeated for a limited number of iterations or until the result meets the accuracy requirements

5. **Result Export and Visualization**
   - Extracted tables are saved in the specified output format (CSV/JSON)
   - Annotated images showing detected tables are saved for visual inspection
   - All outputs are organized in per-PDF subfolders for easy management

### Extracting Table Data from Images using TableTransformerForObjectDetection

This notebook demonstrates how to use the Transformer-based Table Structure Recognition (TSR) model, `TableTransformerForObjectDetection`, to extract structured table data from images.
- [TableTransformer ](https://huggingface.co/spaces/SalML/TableTransformer2CSV/blob/main/app.py)

We will cover:
1. The meaning of each output from the structure recognition model.
2. The step-by-step process to reconstruct table data from the model's detection results.

#### 1. Understanding the Output of TableTransformerForObjectDetection (Structure Recognition)

When you run an image through the TableTransformerForObjectDetection model for structure recognition, you typically get the following outputs after post-processing:

- **probas**: A tensor of shape [N, num_classes], where N is the number of detected objects (table rows, columns, etc.). Each row is the softmax probability for each class.
- **bboxes_scaled**: A tensor of shape [N, 4], where each row is [xmin, ymin, xmax, ymax] representing the bounding box coordinates of a detected object, mapped to the original image size.
- **model.config.id2label**: A mapping from class indices to human-readable labels (e.g., 'table row', 'table column').

For each detected object (row/column):
- The class with the highest probability in `probas` indicates the type of the object (row, column, etc.).
- The corresponding bounding box in `bboxes_scaled` gives the location of that object in the image.

#### 2. Step-by-Step Table Reconstruction from Detection Results

The following steps describe how to reconstruct the table structure and extract cell data from the model's output:

##### Step 1: Separate Rows and Columns
- Use the class labels to group bounding boxes into rows and columns.
- Sort rows by their vertical (y) position, and columns by their horizontal (x) position.

##### Step 2: Crop Each Row and Then Each Cell
- For each row bounding box, crop the row region from the table image.
- For each row, use the column bounding boxes to further crop out each cell.
- The intersection of a row and a column bounding box gives the cell region.

##### Step 3: OCR Each Cell
- Use an OCR engine (e.g., Tesseract) to extract text from each cell image.

##### Step 4: Assemble the Table
- Combine the recognized cell texts into a 2D array or pandas DataFrame, using the row and column order.


## Parameter & Data Flow
- User parameters collected via GUI, passed through pipeline
- All output/log/preview organized by output_path
- Logging and exception handling throughout
