# Transformer表格处理逻辑

本文档详细说明PDF Table Extractor中使用Transformer+EasyOCR进行表格识别和处理的完整流程。

## 概述

对于基于图像的PDF（扫描文档），系统使用Transformer模型进行表格检测和结构识别，结合EasyOCR进行文本提取，最终生成结构化的表格数据。

## 处理流程

### 1. 整体处理流程

```
扫描PDF页面 → 图像预处理 → Transformer检测 → 结构识别 → EasyOCR文本提取 → 文本匹配 → 单元格处理 → 生成DataFrame
```

### 2. 核心组件

- **Detection模型**：检测表格区域和单元格边界
- **Structure模型**：识别表格结构和单元格关系
- **EasyOCR**：提取文本内容
- **TableParser**：处理识别结果，生成最终表格

## 详细实现

### 1. 主处理流程

**方法位置**：`core/models/table_parser.py` - `TableParser.parse_table()`

```python
def parse_table(self, image, table_bbox=None, save_visualization=False):
    """
    主处理流程：检测 → 识别 → 文本提取 → 匹配 → 生成表格
    
    Args:
        image: 输入图像
        table_bbox: 表格边界框（可选）
        save_visualization: 是否保存可视化结果
    
    Returns:
        dict: 包含success, error, tables等键的结果字典
    """
    try:
        # 1. 图像预处理
        processed_image = self._preprocess_image(image)
        
        # 2. 表格检测
        detection_results = self._detect_tables(processed_image, table_bbox)
        
        # 3. 结构识别
        structure_results = self._recognize_structure(processed_image, detection_results)
        
        # 4. 文本提取
        ocr_results = self._extract_text_with_ocr(processed_image)
        
        # 5. 文本匹配和单元格填充
        filled_tables = self._fill_cell_text(structure_results, ocr_results)
        
        # 6. 生成最终表格
        final_tables = self._generate_final_tables(filled_tables)
        
        # 7. 保存可视化结果（可选）
        if save_visualization:
            self._save_visualization_results(processed_image, detection_results, 
                                           structure_results, ocr_results)
        
        return {
            'success': True,
            'tables': final_tables,
            'error': ''
        }
        
    except Exception as e:
        return {
            'success': False,
            'tables': [],
            'error': str(e)
        }
```

### 2. 表格检测

**方法位置**：`core/models/table_parser.py` - `TableParser._detect_tables()`

```python
def _detect_tables(self, image, table_bbox=None):
    """使用Detection模型检测表格区域"""
    try:
        # 如果指定了表格区域，直接使用
        if table_bbox:
            return [{'bbox': table_bbox, 'confidence': 1.0}]
        
        # 使用Detection模型检测
        detection_results = self.models.detection_model.predict(image)
        
        # 过滤低置信度结果
        filtered_results = []
        for result in detection_results:
            if result['confidence'] > self.detection_threshold:
                filtered_results.append(result)
        
        return filtered_results
        
    except Exception as e:
        self.logger.error(f"Table detection failed: {e}")
        return []
```

### 3. 结构识别

**方法位置**：`core/models/table_parser.py` - `TableParser._recognize_structure()`

```python
def _recognize_structure(self, image, detection_results):
    """使用Structure模型识别表格结构"""
    structure_results = []
    
    for detection in detection_results:
        try:
            # 提取表格区域
            table_bbox = detection['bbox']
            table_crop = self._crop_table_region(image, table_bbox)
            
            # 使用Structure模型识别
            structure_result = self.models.structure_model.predict(table_crop)
            
            # 处理识别结果
            processed_structure = self._process_structure_result(structure_result, table_bbox)
            structure_results.append(processed_structure)
            
        except Exception as e:
            self.logger.error(f"Structure recognition failed: {e}")
            continue
    
    return structure_results
```

### 4. 文本提取

**方法位置**：`core/models/table_parser.py` - `TableParser._extract_text_with_ocr()`

```python
def _extract_text_with_ocr(self, image):
    """使用EasyOCR提取文本内容"""
    try:
        # 初始化EasyOCR Reader
        reader = get_easyocr_reader(['en'])
        
        # 执行OCR
        ocr_results = reader.readtext(image)
        
        # 处理OCR结果
        processed_results = []
        for result in ocr_results:
            bbox, text, confidence = result
            if confidence > self.ocr_threshold:
                processed_results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence
                })
        
        return processed_results
        
    except Exception as e:
        self.logger.error(f"OCR extraction failed: {e}")
        return []
```

### 5. 文本匹配和单元格填充

**方法位置**：`core/models/table_parser.py` - `TableParser._fill_cell_text()`

```python
def _fill_cell_text(self, structure_results, ocr_results):
    """将OCR文本匹配到单元格中"""
    filled_tables = []
    
    for structure in structure_results:
        try:
            # 获取单元格坐标
            cell_coordinates = structure['cell_coordinates']
            
            # 创建文本匹配映射
            cell_text_map = {}
            
            # 按行聚类OCR文本
            text_rows = self._cluster_ocr_texts_by_rows(ocr_results)
            
            # 匹配文本到单元格
            for row_idx, row_cells in enumerate(cell_coordinates):
                for col_idx, cell_bbox in enumerate(row_cells):
                    cell_key = (row_idx, col_idx)
                    
                    # 查找匹配的文本
                    matched_texts = self._find_matching_texts(cell_bbox, text_rows)
                    
                    # 合并文本内容
                    cell_text = self._merge_cell_texts(matched_texts)
                    cell_text_map[cell_key] = cell_text
            
            # 处理跨行/跨列单元格
            cell_text_map = self._handle_spanning_cells(cell_text_map, structure)
            
            filled_tables.append({
                'structure': structure,
                'cell_text_map': cell_text_map
            })
            
        except Exception as e:
            self.logger.error(f"Cell text filling failed: {e}")
            continue
    
    return filled_tables
```

## 结构识别结果处理

### 1. Detection模型输出

Detection模型返回的结果包含：
- **bbox**：表格边界框坐标 `[x1, y1, x2, y2]`
- **confidence**：检测置信度
- **class_id**：表格类型标识

### 2. Structure模型输出

Structure模型返回的结果包含：
- **cell_bboxes**：所有单元格的边界框
- **row_headers**：行标题信息
- **column_headers**：列标题信息
- **spanning_cells**：跨行/跨列单元格信息

**详细说明**：参考 [结构识别结果分析](recognize_structure_analysis.md)

### 3. 单元格坐标计算

```python
def _calculate_cell_coordinates(self, structure_result):
    """计算单元格坐标"""
    cell_coordinates = []
    
    # 按行分组单元格
    rows = defaultdict(list)
    for cell in structure_result['cell_bboxes']:
        row_idx = cell['row']
        rows[row_idx].append(cell)
    
    # 按行排序
    for row_idx in sorted(rows.keys()):
        row_cells = sorted(rows[row_idx], key=lambda x: x['column'])
        cell_coordinates.append(row_cells)
    
    return cell_coordinates
```

## 文本匹配算法

### 1. OCR文本聚类

```python
def _cluster_ocr_texts_by_rows(self, ocr_results):
    """将OCR文本按行聚类"""
    # 按y坐标排序
    sorted_ocr = sorted(ocr_results, key=lambda x: x['bbox'][0][1])
    
    text_rows = defaultdict(list)
    current_row = 0
    
    for i, ocr_result in enumerate(sorted_ocr):
        if i == 0:
            text_rows[current_row].append(ocr_result)
        else:
            # 计算与上一行的距离
            prev_y = sorted_ocr[i-1]['bbox'][0][1]
            curr_y = ocr_result['bbox'][0][1]
            y_distance = abs(curr_y - prev_y)
            
            # 如果距离大于行高阈值，认为是新行
            if y_distance > self.row_height_threshold:
                current_row += 1
            
            text_rows[current_row].append(ocr_result)
    
    return dict(text_rows)
```

### 2. 文本与单元格匹配

```python
def _find_matching_texts(self, cell_bbox, text_rows):
    """查找与单元格匹配的文本"""
    matched_texts = []
    
    for row_texts in text_rows.values():
        for text in row_texts:
            text_bbox = text['bbox']
            
            # 计算IoU
            iou = self._calculate_iou(cell_bbox, text_bbox)
            
            # 计算包含关系
            is_contained = self._is_text_contained_in_cell(text_bbox, cell_bbox)
            
            # 如果IoU > 阈值或文本被单元格包含
            if iou > self.iou_threshold or is_contained:
                matched_texts.append(text)
    
    return matched_texts
```

### 3. IoU计算

```python
def _calculate_iou(self, bbox1, bbox2):
    """计算两个边界框的IoU"""
    # 计算交集
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0
```

## 跨行/跨列单元格处理

### 1. Spanning Cells识别

```python
def _handle_spanning_cells(self, cell_text_map, structure):
    """处理跨行/跨列单元格"""
    spanning_info = structure.get('spanning_cells', [])
    
    for spanning_cell in spanning_info:
        # 获取主单元格位置
        main_cell = spanning_cell['main_cell']
        covered_cells = spanning_cell['covered_cells']
        
        # 合并覆盖单元格的文本
        merged_text = self._merge_spanning_texts(cell_text_map, main_cell, covered_cells)
        
        # 更新主单元格文本
        cell_text_map[main_cell] = merged_text
        
        # 清空覆盖单元格的文本
        for covered_cell in covered_cells[1:]:  # 跳过主单元格
            cell_text_map[covered_cell] = ""
    
    return cell_text_map
```

### 2. 文本合并

```python
def _merge_spanning_texts(self, cell_text_map, main_cell, covered_cells):
    """合并跨行/跨列单元格的文本"""
    all_texts = []
    
    for cell in covered_cells:
        if cell in cell_text_map and cell_text_map[cell]:
            all_texts.append(cell_text_map[cell])
    
    # 按位置排序并合并
    if all_texts:
        return ' '.join(all_texts)
    
    return ""
```

## 可视化功能

### 1. 检测结果可视化

```python
def _save_detection_visualization(self, image, detection_results, save_path):
    """保存检测结果可视化"""
    vis_image = image.copy()
    
    for detection in detection_results:
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        # 绘制边界框
        cv2.rectangle(vis_image, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        # 绘制置信度
        cv2.putText(vis_image, f"{confidence:.2f}", 
                   (int(bbox[0]), int(bbox[1]) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(save_path, vis_image)
```

### 2. 结构识别可视化

```python
def _save_structure_visualization(self, image, structure_results, save_path):
    """保存结构识别可视化"""
    vis_image = image.copy()
    
    for structure in structure_results:
        cell_coordinates = structure['cell_coordinates']
        
        for row_idx, row_cells in enumerate(cell_coordinates):
            for col_idx, cell_bbox in enumerate(row_cells):
                # 绘制单元格边界
                cv2.rectangle(vis_image, 
                             (int(cell_bbox[0]), int(cell_bbox[1])), 
                             (int(cell_bbox[2]), int(cell_bbox[3])), 
                             (255, 0, 0), 1)
                
                # 绘制行列索引
                cv2.putText(vis_image, f"{row_idx},{col_idx}", 
                           (int(cell_bbox[0]), int(cell_bbox[1]) + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    cv2.imwrite(save_path, vis_image)
```

### 3. 特殊标签可视化

```python
def _save_special_labels_visualization(self, image, ocr_results, save_path):
    """保存特殊标签可视化"""
    vis_image = image.copy()
    
    for ocr_result in ocr_results:
        bbox = ocr_result['bbox']
        text = ocr_result['text']
        confidence = ocr_result['confidence']
        
        # 绘制文本边界框
        cv2.rectangle(vis_image, 
                     (int(bbox[0][0]), int(bbox[0][1])), 
                     (int(bbox[2][0]), int(bbox[2][1])), 
                     (0, 0, 255), 1)
        
        # 绘制文本内容
        cv2.putText(vis_image, f"{text} ({confidence:.2f})", 
                   (int(bbox[0][0]), int(bbox[0][1]) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imwrite(save_path, vis_image)
```

## 最终表格生成

### 1. DataFrame生成

```python
def _generate_final_tables(self, filled_tables):
    """生成最终的DataFrame表格"""
    final_tables = []
    
    for table_data in filled_tables:
        try:
            structure = table_data['structure']
            cell_text_map = table_data['cell_text_map']
            
            # 获取表格尺寸
            num_rows = len(structure['cell_coordinates'])
            num_cols = len(structure['cell_coordinates'][0]) if structure['cell_coordinates'] else 0
            
            # 创建DataFrame
            df_data = []
            for row_idx in range(num_rows):
                row_data = []
                for col_idx in range(num_cols):
                    cell_key = (row_idx, col_idx)
                    cell_text = cell_text_map.get(cell_key, "")
                    row_data.append(cell_text)
                df_data.append(row_data)
            
            # 创建DataFrame
            df = pd.DataFrame(df_data)
            
            final_tables.append({
                'dataframe': df,
                'bbox': structure['table_bbox'],
                'metadata': {
                    'num_rows': num_rows,
                    'num_cols': num_cols,
                    'extraction_method': 'transformer'
                }
            })
            
        except Exception as e:
            self.logger.error(f"DataFrame generation failed: {e}")
            continue
    
    return final_tables
```

## 性能优化

### 1. 模型缓存

```python
class TableModels:
    def __init__(self):
        self._detection_model = None
        self._structure_model = None
    
    @property
    def detection_model(self):
        if self._detection_model is None:
            self._detection_model = self._load_detection_model()
        return self._detection_model
    
    @property
    def structure_model(self):
        if self._structure_model is None:
            self._structure_model = self._load_structure_model()
        return self._structure_model
```

### 2. 并行处理

```python
def _parallel_ocr_processing(self, image_regions):
    """并行处理多个图像区域的OCR"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for region in image_regions:
            future = executor.submit(self._extract_text_with_ocr, region)
            futures.append(future)
        
        results = []
        for future in futures:
            results.extend(future.result())
        
        return results
```

## 注意事项

1. **模型精度**：Transformer模型的精度直接影响表格识别效果
2. **OCR质量**：EasyOCR的文本识别质量影响最终结果
3. **参数调优**：IoU阈值、置信度阈值等参数需要根据实际情况调整
4. **内存管理**：大图像处理时需要注意内存使用

## 后续优化方向

1. **模型优化**：使用更先进的Transformer模型
2. **OCR改进**：集成多种OCR引擎，提高文本识别精度
3. **后处理优化**：改进文本匹配和单元格合并算法
4. **可视化增强**：提供更丰富的可视化选项













