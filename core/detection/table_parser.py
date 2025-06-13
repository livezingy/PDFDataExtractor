# core/detection/table_parser.py
"""
TableParser: Unified table structure recognition and content extraction class
Reference testTransformer.py main process, integrates table structure recognition and OCR content extraction.
"""
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import pandas as pd
import time, asyncio, string, statistics, os,torch
from dataclasses import dataclass
from core.models.table_models import TableModels
from core.utils.logger import AppLogger
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from collections import Counter
from itertools import tee, count
from sympy import im
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
import matplotlib.pyplot as plt
from core.utils.path_utils import get_app_dir
from core.utils.logger import AppLogger


@dataclass
class ParsedTable:
    data: pd.DataFrame
    structure: Any  # Can be a list of structured cells or custom structure
    confidence: float
    bbox: Tuple[float, float, float, float]
    processing_time: float
    metadata: Dict[str, Any] = None


class TableParser:
    """
    Unified table structure recognition and content extraction class
    1. Input the whole page image, detection gets table regions
    2. For each region, crop the image, structure model gets structure
    3. For each cell, OCR, assemble DataFrame
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'min_confidence': 0.5,
            'structure_confidence': 0.5,
            'ocr_confidence': 0.5,
            'enable_data_validation': True
        }
        if config:
            self.config.update(config)
        self.models = TableModels(config)
        self.logger = AppLogger.get_logger()
        self.base_dir = get_app_dir()
        # Ensure Tesseract path is set for all OCR
        import pytesseract
        tesseract_path = os.path.join(self.base_dir, 'models', 'Tesseract-OCR', 'tesseract.exe')
        tessdata_dir = os.path.join(self.base_dir, 'models', 'Tesseract-OCR', 'tessdata')
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        pytesseract.pytesseract.tessdata_dir = tessdata_dir

    def parse_table(self, table_image: Image.Image, bbox: Tuple[float, float, float, float], params: Optional[dict] = None) -> Optional[dict]:
        """
        Input a single table image, return a dict with 'data' and 'columns' for export compatibility.
        Accepts params for structure threshold, border width, preprocess, etc.
        """
        pipeline = TableExtractionPipeline()
        # Use params or fallback to self.config
        structure_threshold = params.get('structure_threshold', self.config.get('structure_confidence', 0.5)) if params else self.config.get('structure_confidence', 0.5)
        border_width = params.get('structure_border_width', 5) if params else 5
        preprocess = params.get('structure_preprocess', True) if params else True
        min_conf = params.get('min_confidence', self.config.get('min_confidence', 0.5)) if params else self.config.get('min_confidence', 0.5)
        # Await the coroutine and get result synchronously
        tables = asyncio.run(pipeline.start_process(
            input_Image=table_image,
            TSR_THRESHOLD=structure_threshold,
            padd_top=border_width, padd_left=border_width, padd_bottom=border_width, padd_right=border_width,
            delta_xmin=5, delta_ymin=5, delta_xmax=5, delta_ymax=5,
            expand_rowcol_bbox_top=5, expand_rowcol_bbox_bottom=5,
            preprocess=preprocess
        ))
        if not tables:
            return None
        table = tables[0]
        if isinstance(table, pd.DataFrame):
            return {
                'data': table.to_dict('records'),
                'columns': table.columns.tolist(),
                'confidence': 1.0,
                'bbox': bbox
            }
        elif isinstance(table, dict) and 'data' in table and 'columns' in table:
            return table
        elif hasattr(table, 'data') and hasattr(table, 'columns'):
            return {
                'data': table.data,
                'columns': table.columns,
                'confidence': getattr(table, 'confidence', 1.0),
                'bbox': bbox
            }
        return None
        


class TableExtractionPipeline():
    # Color list for visualization
    colors = ["red", "blue", "green", "yellow", "orange", "violet"]

    def add_padding(self, pil_img, top, right, bottom, left, color=(255,255,255)):
        '''Add border to the image to avoid losing table edge information.'''
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        AppLogger.get_logger().debug(f"Added padding: top={top}, right={right}, bottom={bottom}, left={left}")
        return result

    def generate_structure(self, model, pil_img, prob, boxes, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        '''Draw structure recognition results (row/column boxes) on the table image and return row/column bounding box info.'''
        
        rows = {}
        cols = {}
        idx = 0
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax 
            cl = p.argmax()
            class_text = model.config.id2label[cl.item()]
            # Save row/column bounding boxes
            if class_text == 'table row':
                rows['table row.'+str(idx)] = (xmin, ymin-expand_rowcol_bbox_top, xmax, ymax+expand_rowcol_bbox_bottom)
            if class_text == 'table column':
                cols['table column.'+str(idx)] = (xmin, ymin-expand_rowcol_bbox_top, xmax, ymax+expand_rowcol_bbox_bottom)
            idx += 1
        AppLogger.get_logger().debug(f"Generated structure: {len(rows)} rows, {len(cols)} columns")
        return rows, cols

    def sort_table_featuresv2(self, rows:dict, cols:dict):
        # Sort rows by ymin, columns by xmin
        rows_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(rows.items(), key=lambda tup: tup[1][1])}
        cols_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])}
        AppLogger.get_logger().debug(f"Sorted table features.")
        return rows_, cols_

    def individual_table_featuresv2(self, pil_img, rows:dict, cols:dict):
        '''Crop each row and column image according to bounding boxes.'''
        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            rows[k] = xmin, ymin, xmax, ymax, cropped_img
        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cols[k] = xmin, ymin, xmax, ymax, cropped_img
        AppLogger.get_logger().debug(f"Cropped individual row and column images.")
        return rows, cols

    def object_to_cellsv2(self, master_row:dict, cols:dict, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left):
        '''Split each row into cell images by columns, return all cell images and table shape.'''
        cells_img = {}
        row_idx = 0
        new_cols = cols
        new_master_row = master_row
        for k_row, v_row in new_master_row.items():
            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            row_img_list = []
            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                xa = xmin_col
                xb = xmax_col
                if idx == 0:
                    xa = 0
                if idx == len(new_cols)-1:
                    xb = xmax
                # Boundary check to avoid right < left
                if xa >= xb or xa < 0 or xb > xmax:
                    continue
                row_img_cropped = row_img.crop((xa, 0, xb, ymax))
                row_img_list.append(row_img_cropped)
            cells_img[k_row+'.'+str(row_idx)] = row_img_list
            row_idx += 1
        AppLogger.get_logger().debug(f"Split rows into cells: {len(cells_img)} rows.")
        return cells_img, len(new_cols), len(new_master_row)-1

    def clean_dataframe(self, df):
        # Remove unwanted characters from DataFrame
        for col in df.columns:
            df[col]=df[col].str.replace("'", '', regex=True)
            df[col]=df[col].str.replace('"', '', regex=True)
            df[col]=df[col].str.replace(r'\]', '', regex=True)
            df[col]=df[col].str.replace(r'\[', '', regex=True)
            df[col]=df[col].str.replace('{', '', regex=True)
            df[col]=df[col].str.replace('}', '', regex=True)
        AppLogger.get_logger().debug(f"Cleaned DataFrame columns: {df.columns.tolist()}")
        return df

    def create_dataframe(self, cells_pytess_result:list, max_cols:int, max_rows:int):
        # Assemble DataFrame from OCR results
        headers = cells_pytess_result[:max_cols]
        new_headers = TableParserUtils.uniquify(headers, (f' {x!s}' for x in string.ascii_lowercase))
        cells_list = cells_pytess_result[max_cols:]
        expected_cells = max_cols * max_rows
        # Defensive: if OCR cell count is not as expected, pad or truncate
        if len(cells_list) < expected_cells:
            AppLogger.get_logger().debug(f"Cell count ({len(cells_list)}) less than expected ({expected_cells}), padding with empty strings.")
            cells_list += [''] * (expected_cells - len(cells_list))
        elif len(cells_list) > expected_cells:
            AppLogger.get_logger().debug(f"Cell count ({len(cells_list)}) greater than expected ({expected_cells}), truncating.")
            cells_list = cells_list[:expected_cells]
        df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)
        cell_idx = 0
        for nrows in range(max_rows):
            for ncols in range(max_cols):
                df.iat[nrows, ncols] = str(cells_list[cell_idx])
                cell_idx += 1
        AppLogger.get_logger().debug(f"Created DataFrame with shape: {df.shape}")
        df = self.clean_dataframe(df)
        return df

    def enhance_cell_image(self, img: Image.Image) -> Image.Image:
        """Apply a series of image enhancement operations before OCR if enabled."""
        # You can add more enhancement steps here as needed
        #img = TableParserUtils.super_res(img)
        img = TableParserUtils.sharpen_image(img)
        img = TableParserUtils.binarizeBlur_image(img)
        return img

    async def start_process(self, input_Image: Image.Image, TSR_THRESHOLD, padd_top, padd_left, padd_bottom, padd_right, delta_xmin, delta_ymin, delta_xmax, delta_ymax, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, preprocess=True):
        AppLogger.get_logger().debug("Starting table extraction pipeline.")
        # Add padding, avoid edge information missing
        table = self.add_padding(input_Image, padd_top, padd_right, padd_bottom, padd_left)
        # recognize table structure
        model, probas, bboxes_scaled = TableParserUtils.table_struct_recog(table, THRESHOLD_PROBA=TSR_THRESHOLD, base_dir=get_app_dir())
        AppLogger.get_logger().debug(f"Structure model output: {len(probas)} candidates.")
        rows, cols = self.generate_structure(model, table, probas, bboxes_scaled, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom)
        rows, cols = self.sort_table_featuresv2(rows, cols)
        master_row, cols = self.individual_table_featuresv2(table, rows, cols)
        cells_img, max_cols, max_rows = self.object_to_cellsv2(master_row, cols, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left)
        sequential_cell_img_list = []
        for k, img_list in cells_img.items():
            for img in img_list:
                if preprocess:
                    img = self.enhance_cell_image(img)
                sequential_cell_img_list.append(TableParserUtils.pytess(img))
        AppLogger.get_logger().debug(f"Performing OCR on {len(sequential_cell_img_list)} cells.")
        cells_pytess_result = await asyncio.gather(*sequential_cell_img_list)
       
        df = self.create_dataframe(cells_pytess_result, max_cols, max_rows)
        
        AppLogger.get_logger().debug("Table extraction pipeline finished.")
        return [df]
        


        
class TableParserUtils:
    @staticmethod
    def PIL_to_cv(pil_img):
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv_to_PIL(cv_img):
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    @staticmethod
    async def pytess(cell_pil_img):
        return ' '.join(pytesseract.image_to_data(cell_pil_img, output_type=Output.DICT, config='-c tessedit_char_blacklist=œ˜â€œï¬â™Ã©œ¢!|”?«“¥ --psm 6 preserve_interword_spaces')['text']).strip()

    @staticmethod
    def sharpen_image(pil_img):
        img = TableParserUtils.PIL_to_cv(pil_img)
        sharpen_kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(img, -1, sharpen_kernel)
        pil_img = TableParserUtils.cv_to_PIL(sharpen)
        return pil_img

    @staticmethod
    def uniquify(seq, suffs = count(1)):
        not_unique = [k for k,v in Counter(seq).items() if v>1]
        suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
        for idx,s in enumerate(seq):
            try:
                suffix = str(next(suff_gens[s]))
            except KeyError:
                continue
            else:
                seq[idx] += suffix
        return seq

    @staticmethod
    def binarizeBlur_image(pil_img):
        image = TableParserUtils.PIL_to_cv(pil_img)
        thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]
        result = cv2.GaussianBlur(thresh, (5,5), 0)
        result = 255 - result
        return TableParserUtils.cv_to_PIL(result)

    @staticmethod
    def td_postprocess(pil_img):
        img = TableParserUtils.PIL_to_cv(pil_img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255))
        nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255))
        nzmask = cv2.erode(nzmask, np.ones((3,3)))
        mask = mask & nzmask
        new_img = img.copy()
        new_img[np.where(mask)] = 255
        return TableParserUtils.cv_to_PIL(new_img)   

    @staticmethod
    def table_struct_recog(image, THRESHOLD_PROBA, base_dir):
        from transformers import DetrImageProcessor, TableTransformerForObjectDetection
        feature_extractor = DetrImageProcessor(do_resize=True, size={"height": 1000, "width": 1000})
        encoding = feature_extractor(image, return_tensors="pt")
        structure_model_path = os.path.join(base_dir, 'models', 'table-transformer', 'structure')
        model = TableTransformerForObjectDetection.from_pretrained(structure_model_path)
        with torch.no_grad():
            outputs = model(**encoding)
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > THRESHOLD_PROBA# Confidence threshold
        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
        postprocessed_outputs = feature_extractor.post_process_object_detection(outputs, threshold=0., target_sizes=target_sizes)
        bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
        return (model, probas[keep], bboxes_scaled)


