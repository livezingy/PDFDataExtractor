# 表格可视化模块使用说明

## 概述

`TableVisualize` 模块提供了完整的表格检测和结构可视化功能，包括：

1. **模型检测可视化** - 显示检测到的表格区域
2. **单元格检测可视化** - 显示单元格边界和坐标
3. **表格结构可视化** - 显示表格结构，包括合并单元格

## 基本使用

### 1. 导入模块

```python
from core.models.table_visualize import TableVisualize
from PIL import Image
```

### 2. 初始化可视化器

```python
visualizer = TableVisualize()
```

### 3. 表格结构可视化

```python
# 准备表格数据
table_data = {
    'table_rows': [
        {'bbox': [50, 50, 400, 80]},
        {'bbox': [50, 80, 400, 110]},
        # ... 更多行
    ],
    'table_cols': [
        {'bbox': [50, 50, 150, 170]},
        {'bbox': [150, 50, 250, 170]},
        # ... 更多列
    ],
    'special_labels': {
        'column_headers': [
            {'bbox': [50, 50, 150, 80], 'label': 'table column header'},
            # ... 更多列标题
        ],
        'projected_row_headers': [
            {'bbox': [50, 80, 150, 110], 'label': 'table projected row header'},
            # ... 更多行标题
        ],
        'spanning_cells': [
            {
                'bbox': [150, 80, 350, 110], 
                'label': 'table spanning cell',
                'col_span': 2,
                'row_span': 1,
                'span_type': 'column'
            },
            # ... 更多合并单元格
        ]
    }
}

# 执行可视化
table_image = Image.open("path/to/table.png")
visualizer.visualize_table_structure(table_image, table_data, "output.png")
```

### 4. 单元格检测可视化

```python
# 准备单元格坐标数据
cell_coordinates = [
    {
        'row': [50, 50, 400, 80],
        'cells': [
            {'column': [50, 50, 150, 80], 'cell': [50, 50, 150, 80]},
            {'column': [150, 50, 250, 80], 'cell': [150, 50, 250, 80]},
            # ... 更多单元格
        ],
        'cell_count': 4
    },
    # ... 更多行
]

# 执行可视化
visualizer.visualize_cell_detection(table_image, cell_coordinates, "cells.png")
```

### 5. 综合可视化

```python
# 一次性生成所有类型的可视化
saved_files = visualizer.create_comprehensive_visualization(
    table_image, 
    table_data, 
    cell_coordinates,
    save_dir="output"
)

print("生成的文件:")
for key, path in saved_files.items():
    print(f"  {key}: {path}")
```

## 可视化特性

### 颜色编码

- **蓝色** - 表格行
- **绿色** - 表格列
- **红色** - 列标题
- **橙色** - 行标题
- **紫色** - 行列合并单元格
- **青色** - 列合并单元格
- **洋红色** - 行合并单元格

### 合并单元格支持

可视化模块完全支持合并单元格的显示：

- `span_type: 'both'` - 行列都合并，使用紫色和斜线填充
- `span_type: 'column'` - 仅列合并，使用青色和竖线填充
- `span_type: 'row'` - 仅行合并，使用洋红色和横线填充

### 图例说明

每个可视化图像都包含详细的图例，说明不同颜色和样式的含义。

## 与TableParser集成

可视化模块设计为与 `TableParser` 无缝集成：

```python
from core.models.table_parser import TableParser
from core.models.table_visualize import TableVisualize

# 解析表格
parser = TableParser(config)
result = await parser.parser_image(image)

# 可视化结果
visualizer = TableVisualize()
for table in result['tables']:
    # 提取表格数据并可视化
    table_data = extract_table_data(table)
    visualizer.visualize_table_structure(table_image, table_data, "output.png")
```

## 注意事项

1. 确保输入图像为RGB格式
2. 边界框坐标格式为 `[x1, y1, x2, y2]`
3. 合并单元格需要提供 `col_span` 和 `row_span` 信息
4. 可视化文件默认保存为PNG格式，300 DPI

