# 参数计算公式说明文档

本文档详细说明了pdfplumber和Camelot（lattice + stream）的参数计算公式，包括所有统计量的含义和计算方法。

---

## 一、页面特征统计量说明

### 1. 字符特征（char_analysis）

| 统计量 | 含义 | 说明 |
|--------|------|------|
| `total_chars` | 字符总数 | 当前页面中所有字符的数量 |
| `min_width` | 最小字符宽度 | 当前页面中所有字符宽度的最小值（单位：点pt） |
| `min_height` | 最小字符高度 | 当前页面中所有字符高度的最小值（单位：点pt） |
| `max_width` | 最大字符宽度 | 当前页面中所有字符宽度的最大值（单位：点pt） |
| `max_height` | 最大字符高度 | 当前页面中所有字符高度的最大值（单位：点pt） |
| `mode_width` | 字符宽度众数 | 当前页面中所有字符宽度出现次数最多的值（单位：点pt） |
| `mode_height` | 字符高度众数 | 当前页面中所有字符高度出现次数最多的值（单位：点pt） |

**计算说明**：
- 如果数据不足（<3个字符），众数回退到最小值
- 如果所有值都不同（无重复），众数回退到最小值

### 2. 文本行特征（text_line_analysis）

| 统计量 | 含义 | 说明 |
|--------|------|------|
| `total_lines` | 文本行总数 | 当前页面中所有文本行的数量 |
| `min_line_height` | 最小行高 | 当前页面中所有文本行高度的最小值（单位：点pt） |
| `max_line_height` | 最大行高 | 当前页面中所有文本行高度的最大值（单位：点pt） |
| `mode_line_height` | 行高众数 | 当前页面中所有文本行高度出现次数最多的值（单位：点pt） |
| `min_line_spacing` | 最小行间距 | 相邻文本行之间的最小间距（单位：点pt） |
| `max_line_spacing` | 最大行间距 | 相邻文本行之间的最大间距（单位：点pt） |
| `mode_line_spacing` | 行间距众数 | 相邻文本行之间间距出现次数最多的值（单位：点pt） |

**计算说明**：
- 行高 = 文本行的 bottom - top
- 行间距 = 下一行的 top - 当前行的 bottom（只计算正值）

### 3. 线条特征（line_analysis）

| 统计量 | 含义 | 说明 |
|--------|------|------|
| `horizontal_lines` | 水平线条列表 | 当前页面中所有水平线条的对象列表 |
| `vertical_lines` | 垂直线条列表 | 当前页面中所有垂直线条的对象列表 |
| `horizontal_lines_length` | 水平线长度列表 | 所有水平线条的长度列表（单位：点pt） |
| `vertical_lines_length` | 垂直线长度列表 | 所有垂直线条的长度列表（单位：点pt） |
| `line_widths` | 线条宽度列表 | 所有线条的宽度列表（单位：点pt） |
| `min_horizontal_length` | 最小水平线长度 | 所有水平线条长度的最小值（单位：点pt） |
| `max_horizontal_length` | 最大水平线长度 | 所有水平线条长度的最大值（单位：点pt） |
| `mode_horizontal_length` | 水平线长度众数 | 所有水平线条长度出现次数最多的值（单位：点pt） |
| `min_vertical_length` | 最小垂直线长度 | 所有垂直线条长度的最小值（单位：点pt） |
| `max_vertical_length` | 最大垂直线长度 | 所有垂直线条长度的最大值（单位：点pt） |
| `mode_vertical_length` | 垂直线长度众数 | 所有垂直线条长度出现次数最多的值（单位：点pt） |
| `min_line_width` | 最小线条宽度 | 所有线条宽度的最小值（单位：点pt） |
| `max_line_width` | 最大线条宽度 | 所有线条宽度的最大值（单位：点pt） |
| `mode_line_width` | 线条宽度众数 | 所有线条宽度出现次数最多的值（单位：点pt） |

**计算说明**：
- 水平线长度 = abs(x1 - x0)
- 垂直线长度 = abs(y1 - y0)
- 线条宽度从 `linewidth` 属性获取，如果没有则使用默认值

### 4. 单词特征（word_analysis）

| 统计量 | 含义 | 说明 |
|--------|------|------|
| `total_words` | 单词总数 | 当前页面中所有单词的数量 |
| `avg_words_per_line` | 平均每行单词数 | 平均每行包含的单词数量（仅用于统计，不用于参数计算） |
| `avg_word_spacing` | 平均单词间距 | 相邻单词之间的平均间距（仅用于统计，不用于参数计算） |
| `words_per_line_distribution` | 每行单词数分布 | 每行单词数量的列表，用于计算分位数 |

---

## 二、pdfplumber 参数计算公式

### 1. snap_tolerance（线条合并容差）

**公式**：
```
snap_tolerance = min(min_width, min_height) × 0.3
```

**参数范围**：`[0.5, 5]`

**说明**：
- 使用最小字符尺寸的30%作为容差
- 用于合并接近的平行线条
- 保守策略，避免错误合并

**示例**：
- 如果 `min_width = 4pt`, `min_height = 5pt`
- 则 `snap_tolerance = min(4, 5) × 0.3 = 4 × 0.3 = 1.2pt`
- 最终限制在 `[0.5, 5]` 范围内

---

### 2. join_tolerance（线段连接容差）

**公式**：
```
join_tolerance = min(min_width, min_height) × 0.3
```

**参数范围**：`[1, 10]`

**说明**：
- 与 `snap_tolerance` 相同的基础计算
- 用于连接同一无限直线上、端点接近的线段
- 保守策略，避免错误连接

**示例**：
- 如果 `min_width = 4pt`, `min_height = 5pt`
- 则 `join_tolerance = min(4, 5) × 0.3 = 4 × 0.3 = 1.2pt`
- 最终限制在 `[1, 10]` 范围内

---

### 3. edge_min_length（边缘最小长度）

**公式**：
```
edge_min_length = max(mode_width, mode_height)
```

**回退策略**：
- 如果众数不可用，回退到：`max(min_width, min_height)`

**参数范围**：`[1, 30]`

**说明**：
- 使用字符尺寸的众数作为最小边缘长度
- 众数能更好地反映表格的实际字符尺寸
- 避免装饰性线条或异常值的影响

**示例**：
- 如果 `mode_width = 6pt`, `mode_height = 8pt`
- 则 `edge_min_length = max(6, 8) = 8pt`
- 最终限制在 `[1, 30]` 范围内

---

### 4. intersection_tolerance（交点容差）

**公式**：
```
intersection_tolerance = max(mode_width, mode_height) × 0.5
```

**回退策略**：
- 如果众数不可用，回退到：`max(min_width, min_height) × 0.5`

**参数范围**：`[1, 10]`

**说明**：
- 使用字符尺寸众数的50%作为交点容差
- 用于判断正交边缘是否相交
- 保守策略，避免错误判断交点

**示例**：
- 如果 `mode_width = 6pt`, `mode_height = 8pt`
- 则 `intersection_tolerance = max(6, 8) × 0.5 = 8 × 0.5 = 4pt`
- 最终限制在 `[1, 10]` 范围内

---

### 5. min_words_vertical（垂直方向最小单词数）

**公式**：
```
min_words_vertical = max(3, min(int(total_lines × 0.2), 10))
```

**参数范围**：`[3, 10]`

**说明**：
- 基于文本行总数的20%计算
- 用于垂直策略（text模式）时，要求至少这么多单词对齐
- 限制在合理范围内

**示例**：
- 如果 `total_lines = 15`
- 则 `min_words_vertical = max(3, min(int(15 × 0.2), 10)) = max(3, min(3, 10)) = 3`
- 如果 `total_lines = 50`
- 则 `min_words_vertical = max(3, min(int(50 × 0.2), 10)) = max(3, min(10, 10)) = 10`

---

### 6. min_words_horizontal（水平方向最小单词数）

**公式**：
```
p10_words = percentile(words_per_line_distribution, 10)
min_words_horizontal = max(1, int(p10_words))
```

**参数范围**：`[1, 5]`

**说明**：
- 基于每行单词数分布的第10百分位数
- 用于水平策略（text模式）时，要求至少这么多单词对齐
- 保守策略，避免要求过高

**示例**：
- 如果 `words_per_line_distribution = [3, 4, 5, 5, 6, 7, 8]`
- 则 `p10_words = percentile([3, 4, 5, 5, 6, 7, 8], 10) ≈ 3.0`
- 则 `min_words_horizontal = max(1, int(3.0)) = 3`
- 最终限制在 `[1, 5]` 范围内

---

### 7. text_x_tolerance（文本X方向容差）

**公式**：
```
text_x_tolerance = mode_width × 1.5
```

**回退策略**：
- 如果众数不可用，回退到：`min_width × 1.5`

**参数范围**：`[1, 10]`

**说明**：
- 使用字符宽度众数的1.5倍作为X方向容差
- 用于文本模式时，判断单词内字符的距离
- 保守策略，避免错误分割单词

**示例**：
- 如果 `mode_width = 6pt`
- 则 `text_x_tolerance = 6 × 1.5 = 9pt`
- 最终限制在 `[1, 10]` 范围内

---

### 8. text_y_tolerance（文本Y方向容差）

**公式**：
```
text_y_tolerance = min_line_height × 0.2
```

**参数范围**：`[1, 8]`

**说明**：
- 使用最小行高的20%作为Y方向容差
- 用于文本模式时，判断单词内字符的距离
- 保守策略，避免错误分割单词

**示例**：
- 如果 `min_line_height = 10pt`
- 则 `text_y_tolerance = 10 × 0.2 = 2pt`
- 最终限制在 `[1, 8]` 范围内

---

## 三、Camelot Lattice 参数计算公式

### 1. line_scale（线条缩放因子）

**公式**：
```
# 步骤1: 计算线条宽度众数
mode_line_width = mode(line_widths)  # 如果数据不足，回退到min(line_widths)

# 步骤2: 计算PDF到图像的缩放比例
pdf_to_image_ratio = min(image_height / page_height, image_width / page_width)

# 步骤3: 将PDF坐标的线条宽度转换为图像坐标
mode_line_width_image = mode_line_width × pdf_to_image_ratio

# 步骤4: 计算期望的kernel长度（线条宽度的3倍）
desired_kernel_length = mode_line_width_image × 3

# 步骤5: 计算垂直和水平方向的line_scale
line_scale_v = image_height / desired_kernel_length
line_scale_h = image_width / desired_kernel_length

# 步骤6: 取较小值（保守策略）
line_scale = min(line_scale_v, line_scale_h)

# 步骤7: 限制在合理范围
line_scale = max(15, min(int(line_scale), 50))
```

**参数范围**：`[15, 50]`

**说明**：
- 基于线条宽度众数计算，避免装饰性线条影响
- 考虑PDF到图像的缩放比例
- 使用线条宽度的3倍作为kernel长度
- 取垂直和水平方向计算的较小值，保守策略

**示例**：
- 假设 `mode_line_width = 0.5pt`
- 假设 `image_height = 600px`, `image_width = 800px`
- 假设 `page_height = 800pt`, `page_width = 600pt`
- 则 `pdf_to_image_ratio = min(600/800, 800/600) = min(0.75, 1.33) = 0.75`
- 则 `mode_line_width_image = 0.5 × 0.75 = 0.375px`
- 则 `desired_kernel_length = 0.375 × 3 = 1.125px`
- 则 `line_scale_v = 600 / 1.125 = 533`（超出范围）
- 则 `line_scale_h = 800 / 1.125 = 711`（超出范围）
- 则 `line_scale = min(533, 711) = 533`
- 最终限制在 `[15, 50]` 范围内，结果为 `50`

**注意**：如果计算出的 `desired_kernel_length` 过小，可能导致 `line_scale` 过大。实际计算中，如果 `desired_kernel_length` 接近0或过小，会使用默认值40。

---

### 2. line_tol（线条合并容差）

**公式**：
```
line_tol = min(min_width, min_height) × 0.3
```

**参数范围**：`[0.5, 3]`

**说明**：
- 与 `snap_tolerance` 相同的基础计算
- 用于合并接近的垂直和水平线条
- 保守策略，避免错误合并

**示例**：
- 如果 `min_width = 4pt`, `min_height = 5pt`
- 则 `line_tol = min(4, 5) × 0.3 = 4 × 0.3 = 1.2pt`
- 最终限制在 `[0.5, 3]` 范围内

---

### 3. joint_tol（交点容差）

**公式**：
```
joint_tol = line_tol  # 与line_tol相同
```

**参数范围**：`[0.5, 3]`

**说明**：
- 与 `line_tol` 使用相同的计算公式
- 用于判断线条交点
- 保持一致性，简化计算

**示例**：
- 如果 `line_tol = 1.2pt`
- 则 `joint_tol = 1.2pt`

---

## 四、Camelot Stream 参数计算公式

### 1. edge_tol（文本边缘容差）

**公式**：
```
# 步骤1: 计算最小值
edge_tol_min = min_line_spacing + max_line_height

# 步骤2: 计算最大值
edge_tol_max = page_height / 3

# 步骤3: 计算基础值（使用众数）
if mode_line_spacing > 0 and mode_line_height > 0:
    edge_tol = mode_line_spacing × 3 + mode_line_height × 2
elif mode_line_spacing > 0:
    edge_tol = mode_line_spacing × 3 + max_line_height × 2
else:
    edge_tol = edge_tol_min  # 回退到最小值

# 步骤4: 限制在最小值到最大值之间
edge_tol = max(edge_tol_min, min(edge_tol, edge_tol_max))
```

**参数范围**：`[edge_tol_min, edge_tol_max]`

**说明**：
- 最小值：行间距最小值 + 行高最大值（保守策略）
- 最大值：页面高度的1/3（防止过度连接）
- 基础计算：使用行间距和行高的众数（更准确）
- 最终值限制在最小值和最大值之间

**示例**：
- 假设 `min_line_spacing = 2pt`, `max_line_height = 12pt`
- 假设 `mode_line_spacing = 3pt`, `mode_line_height = 10pt`
- 假设 `page_height = 800pt`
- 则 `edge_tol_min = 2 + 12 = 14pt`
- 则 `edge_tol_max = 800 / 3 = 266.67pt`
- 则 `edge_tol = 3 × 3 + 10 × 2 = 9 + 20 = 29pt`
- 最终限制：`edge_tol = max(14, min(29, 266.67)) = 29pt`

---

### 2. row_tol（行容差）

**公式**：
```
row_tol = min_height
```

**参数范围**：`[1, 10]`

**说明**：
- 使用最小字符高度作为行容差
- 用于垂直组合文本，生成行
- 保守策略，避免错误合并行

**示例**：
- 如果 `min_height = 8pt`
- 则 `row_tol = 8pt`
- 最终限制在 `[1, 10]` 范围内

---

### 3. column_tol（列容差）

**公式**：
```
column_tol = min_width
```

**参数范围**：`[0, 5]`

**说明**：
- 使用最小字符宽度作为列容差
- 用于水平组合文本，生成列
- 保守策略，避免错误合并列

**示例**：
- 如果 `min_width = 4pt`
- 则 `column_tol = 4pt`
- 最终限制在 `[0, 5]` 范围内

---

## 五、统计量计算方法

### 1. 最小值（min）

**方法**：`numpy.min()`

**时间复杂度**：O(n)

**空间复杂度**：O(1)

**说明**：遍历一次，找到最小值

---

### 2. 最大值（max）

**方法**：`numpy.max()`

**时间复杂度**：O(n)

**空间复杂度**：O(1)

**说明**：遍历一次，找到最大值

---

### 3. 众数（mode）

**方法**：`collections.Counter().most_common(1)`

**时间复杂度**：O(n)

**空间复杂度**：O(k)，k为不同值的数量

**回退策略**：
- 如果数据量 < 3，回退到最小值
- 如果所有值都不同（无重复），回退到最小值
- 如果众数出现次数 < 2，回退到最小值

**实现**：
```python
def get_mode_with_fallback(values, min_count=3):
    if not values:
        return 0
    if len(values) < min_count:
        return min(values)
    
    counter = Counter(values)
    most_common = counter.most_common(1)
    
    if most_common and most_common[0][1] >= 2:
        return most_common[0][0]
    else:
        return min(values)
```

---

### 4. 分位数（percentile）

**方法**：`numpy.percentile(values, percentile)`

**时间复杂度**：O(n log n)（需要排序）

**空间复杂度**：O(n)（排序后的数组）

**说明**：用于计算第10百分位数（p10），用于 `min_words_horizontal`

---

## 六、参数计算原则总结

### 1. 容差类参数 → 使用最小值（保守策略）

- `snap_tolerance` = `min(min_width, min_height) × 0.3`
- `join_tolerance` = `min(min_width, min_height) × 0.3`
- `line_tol` = `min(min_width, min_height) × 0.3`
- `joint_tol` = `line_tol`

### 2. 尺寸/长度类参数 → 使用众数（代表性）

- `edge_min_length` = `max(mode_width, mode_height)`
- `text_x_tolerance` = `mode_width × 1.5`
- `line_scale` = 基于 `mode_line_width` 计算

### 3. 阈值类参数 → 使用最小值（保守策略）

- `row_tol` = `min_height`
- `column_tol` = `min_width`
- `text_y_tolerance` = `min_line_height × 0.2`

### 4. 基于分布的参数

- `min_words_horizontal` = 基于 `words_per_line_distribution` 的第10百分位数
- `min_words_vertical` = 基于 `total_lines` 的20%

---

## 七、注意事项

1. **众数回退策略**：如果数据不足或所有值都不同，众数会回退到最小值，确保计算的稳定性。

2. **参数范围限制**：所有参数都限制在合理范围内，避免极端值导致的错误。

3. **保守策略**：大多数参数使用最小值或众数，采用保守策略，避免过度激进导致错误合并或分割。

4. **缩放比例**：Camelot lattice 的 `line_scale` 需要考虑PDF到图像的缩放比例，确保计算的准确性。

5. **默认值**：如果无法计算参数（数据不足），会使用默认值，确保程序的健壮性。

---

## 八、版本信息

- **文档版本**：v2.0
- **更新日期**：2025-12-12
- **主要变更**：
  - 移除均值统计，改用最小值和众数
  - 更新所有参数计算公式
  - 添加回退策略说明
  - 添加详细的统计量含义说明

