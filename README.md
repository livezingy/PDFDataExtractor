# PDFDataExtractor

## Overview
PDFDataExtractor is a robust, extensible pipeline for extracting tables and structured data from PDF documents. It is designed to handle a wide variety of PDF types, including text-based, image-based, and hybrid documents. The system integrates multiple table detection and extraction strategies (Camelot and deep learning-based table-transformer), and provides dynamic parameter optimization to maximize extraction accuracy. The project features a modern GUI built with PySide6, supporting batch processing, parameter customization, and visual preview of extraction results.

## Environment Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Project Structure
The project should have the following directory structure:
```
PDFDataExtractor/
├── main.py
├── models/
│   ├── Tesseract-OCR/
│   │   ├── tesseract.exe
│   │   └── tessdata/
│   └── table-transformer/
│       ├── detection/
│       ├── structure/
│       └── ocr/
└── requirements.txt
```

### 3. Model Files Configuration
The project requires the following model files to be placed in the correct directories:

#### Tesseract-OCR
- Windows: Place the Tesseract-OCR folder in the `models/Tesseract-OCR` directory
  - Ensure it contains `tesseract.exe` and the `tessdata` folder
- Linux: Ensure Tesseract-OCR is installed on the system, default path is `/usr/share/tesseract-ocr/tessdata`

#### Table-Transformer Models
The following three subdirectories are required under `models/table-transformer`:
- `detection/`: Table detection model
  - Download from: [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection)
- `structure/`: Table structure recognition model
  - Download from: [microsoft/table-transformer-structure-recognition](https://huggingface.co/microsoft/table-transformer-structure-recognition)
- `ocr/`: Text recognition model (Currently not used in the codebase, will be added in future updates)
  - Download from: [microsoft/table-transformer-ocr](https://huggingface.co/microsoft/table-transformer-ocr)

### 4. Run the Application
```bash
python main.py
```

## Core Processing Logic

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

## User Interface & Operation

### File Selection
- Use the file panel to add one or more PDF files for processing
- Supports drag & drop and file dialog

### Parameter Configuration
In the parameters panel, select:
- **Camelot Mode**: Choose the extraction flavor (lattice, stream, network, hybrid)
- **Export Format**: Choose CSV or JSON
- **Pages**: Select all pages or specify custom page ranges (e.g., 1-3,5)
- **Output Folder**: Select where results will be saved

### Processing
1. Click "Start Processing" to begin
2. The progress panel displays real-time status, logs, and file processing results
3. If "Preview Detected Tables" is enabled, you can visually inspect detected tables page by page after processing

### Result Review
- Extracted tables and annotated images are saved in the output folder
- Organized by PDF name and type (data, debug, preview)
- Use the Images tab to navigate through detected table previews

## Table Detection Logic

### Camelot vs. Transformer
- **Camelot**: Highly effective for text-based PDFs with clear table lines. Provides detailed table structure and accuracy metrics
- **Table-Transformer**: Robust for image-based or complex PDFs where Camelot may fail or be less accurate

### Decision Strategy
- Prefer Camelot results if accuracy is high and region matches transformer detection
- If Camelot accuracy is low, attempt parameter optimization
- Use transformer+OCR as fallback when necessary
- Always deduplicate and supplement with unmatched high-accuracy Camelot tables

## Advanced Features
- **Thread-Safe Processing**: Each file is processed with isolated parameters for safe concurrent execution
- **Dynamic Parameter Tuning**: Automatic adjustment of extraction parameters based on table structure and quality
- **Comprehensive Logging**: Detailed logs and error tracking for debugging and performance monitoring
- **Extensible Design**: Easily add new detection/extraction modules or customize parameter optimization

## Typical Workflow
1. Launch the app
2. Add PDF files
3. Set output folder and adjust parameters if needed
4. Start processing
5. Monitor progress and review results and previews

## Notes
- For best results, use high-quality, non-encrypted PDFs
- The system is designed for extensibility; you can add new detection/extraction modules or customize parameter optimization as needed
- All output files are organized per PDF for easy management


## Future Plans
1. **Camelot Parameter Optimization**
   - Enhance automatic parameter tuning for Camelot table extraction
   - Implement machine learning-based parameter optimization
   - Add more sophisticated table structure analysis

2. **OCR Model Integration**
   - Integrate the Table-Transformer OCR model
   - Add support for more OCR engines (e.g., EasyOCR, PaddleOCR)
   - Implement OCR model selection based on document characteristics

## More Information
For further details, refer to the code documentation and comments within each module.