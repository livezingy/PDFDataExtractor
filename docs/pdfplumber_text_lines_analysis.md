# PDFPlumber Text Lines 获取与分析

## 📋 概述

本文档说明在pdfplumber中如何获取`text_lines`信息，以及符合什么特征的文本会被认为是`text_lines`。

## 🔍 Text Lines 的获取方式

### 1. 直接属性访问（如果可用）

```python
page = pdf.pages[0]
text_lines = page.text_lines  # 如果pdfplumber版本支持
```

**注意**：并非所有pdfplumber版本都直接提供`text_lines`属性。如果该属性不存在，会返回空列表。

### 2. 从chars构建text_lines（推荐方法）

当`page.text_lines`为空时，可以从`page.chars`手动构建：

```python
def build_text_lines_from_chars(chars, tolerance=2.0):
    """
    从chars构建text_lines
    
    Args:
        chars: page.chars列表
        tolerance: y坐标容差（点）
    
    Returns:
        list: text_lines列表
    """
    if not chars:
        return []
    
    # 按y坐标分组字符
    char_groups = {}
    for char in chars:
        y = char.get('top', 0)  # 使用top作为行的y坐标
        
        # 找到最接近的y坐标组
        matched_y = None
        for group_y in char_groups.keys():
            if abs(y - group_y) <= tolerance:
                matched_y = group_y
                break
        
        if matched_y is None:
            matched_y = y
            char_groups[matched_y] = []
        
        char_groups[matched_y].append(char)
    
    # 构建text_lines
    text_lines = []
    for y, chars in sorted(char_groups.items(), reverse=True):  # 从上到下
        if not chars:
            continue
        
        # 计算行的边界
        tops = [c.get('top', 0) for c in chars]
        bottoms = [c.get('bottom', 0) for c in chars]
        lefts = [c.get('x0', 0) for c in chars]
        rights = [c.get('x1', 0) for c in chars]
        
        text_line = {
            'top': min(tops),
            'bottom': max(bottoms),
            'x0': min(lefts),
            'x1': max(rights),
            'chars': chars
        }
        text_lines.append(text_line)
    
    return text_lines
```

### 3. 使用extract_text_lines()方法（如果可用）

某些pdfplumber版本可能提供`extract_text_lines()`方法：

```python
text_lines = page.extract_text_lines()
```

## 📐 Text Lines 的判断标准

根据pdfplumber的实现原理和字符分组逻辑，符合以下特征的文本会被认为是`text_lines`：

### 1. **垂直位置接近（主要标准）**

- 字符的`top`或`y0`坐标必须在容差范围内（通常2.0点）
- 同一行的字符应该具有相似的垂直位置

```python
# 判断逻辑
if abs(char1['top'] - char2['top']) <= tolerance:
    # 属于同一行
```

### 2. **字符边界框重叠或接近**

- 字符的垂直边界框（`top`到`bottom`）应该有重叠或接近
- 行高通常由该行中字符的最大`bottom`和最小`top`决定

### 3. **水平排列**

- 同一行的字符按`x0`（左边界）从左到右排序
- 字符之间可能有间距，但应该在同一水平线上

### 4. **字体属性（可选）**

- 同一行的字符通常（但不总是）具有相同的字体属性
- 字体大小、字体名称等可以作为辅助判断标准

## 🔧 在当前项目中的实现

### 当前代码位置

`core/processing/page_feature_analyzer.py`:

```python
# 步骤1: 收集所有基础元素
self.text_lines = page.text_lines if hasattr(page, 'text_lines') else []

# 如果text_lines为空，可以从chars构建（需要实现）
if not self.text_lines and self.chars:
    self.text_lines = self._build_text_lines_from_chars()
```

### 为什么text_lines可能为空？

1. **PDF格式问题**：
   - 扫描PDF（图像格式）通常没有文本层，因此没有chars和text_lines
   - 某些PDF的文本可能被编码为路径或图像

2. **无框表格的特殊情况**：
   - 无框表格中的文本可能被pdfplumber识别为表格内容而非普通文本行
   - 表格单元格内的文本可能不会出现在`page.text_lines`中

3. **PDFPlumber版本差异**：
   - 不同版本的pdfplumber对text_lines的支持可能不同
   - 某些版本可能不提供`text_lines`属性

## 💡 解决方案

### 方案1：从chars构建（已实现）

当`page.text_lines`为空时，从`page.chars`构建：

```python
def _build_text_lines_from_chars(self):
    """从chars构建text_lines"""
    # 实现见上面的代码示例
    pass
```

### 方案2：使用layout分析

pdfplumber的layout分析可能提供更准确的文本行信息：

```python
# 使用pdfplumber的layout分析
layout = page.layout
for element in layout:
    if hasattr(element, 'chars'):
        # 处理文本元素
        pass
```

### 方案3：使用extract_text()然后按行分割

```python
text = page.extract_text()
lines = text.split('\n')  # 简单但可能不准确
```

## 📊 Text Lines 的数据结构

每个text_line对象通常包含以下字段：

```python
{
    'top': float,      # 行的顶部y坐标
    'bottom': float,   # 行的底部y坐标
    'x0': float,       # 行的左边界x坐标
    'x1': float,       # 行的右边界x坐标
    'chars': list,     # 该行包含的字符列表（可选）
    'text': str        # 该行的文本内容（可选）
}
```

## ⚠️ 注意事项

1. **容差设置**：
   - y坐标容差（tolerance）需要根据PDF的字体大小调整
   - 太小：可能将同一行分成多行
   - 太大：可能将不同行合并

2. **坐标系**：
   - pdfplumber使用左上角为原点的坐标系
   - y坐标向下为正

3. **性能考虑**：
   - 从chars构建text_lines需要遍历所有字符
   - 对于大文档，可能需要优化算法

## 🔗 参考资源

- [PDFPlumber官方文档](https://github.com/jsvine/pdfplumber)
- [PDFMiner.six文档](https://pdfminersix.readthedocs.io/)（pdfplumber的底层库）

