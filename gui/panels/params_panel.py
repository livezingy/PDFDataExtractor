# gui/panels/params_panel.py
from distutils.command import config
from PySide6.QtWidgets import (QWidget, QFormLayout, QComboBox, QLineEdit, QPushButton, QLabel, QFileDialog,
                              QRadioButton, QButtonGroup, QHBoxLayout, QVBoxLayout,
                              QGroupBox, QSpinBox, QDoubleSpinBox, QTabWidget, QCheckBox)
from PySide6.QtCore import Signal, Qt, QEvent, QPoint
from core.utils.logger import AppLogger
from core.utils.config import Config
import os
import json


class ParamsPanel(QWidget):
    paramsChanged = Signal(dict)
    process_started = Signal(dict)  # Add process_started signal

    def __init__(self, config: Config):
        super().__init__()  # Call base class initialization first
        self.config = config
        
        # 1️⃣ 先获取默认配置
        self.params = self.config.get('ui', {}).copy()
        
        # 2️⃣ 然后初始化UI
        self._init_ui()        
        
        # Ensure panel is visible
        self.setMinimumHeight(300)
        self.setMaximumHeight(500)
        self.setVisible(True)

    def _init_ui(self):
        # Set default control height multiplier
        self._control_height_scale = 1.2
        def set_control_height(widget, base_height=20):
            widget.setFixedHeight(int(base_height * self._control_height_scale))

        # Main layout
        main_layout = QVBoxLayout()

        # --- Tabs ---
        self.tabs = QTabWidget()

        # ========== Basic Parameters Tab ==========
        basic_tab = QWidget()
        basic_form = QFormLayout()
        basic_form.setVerticalSpacing(12)

        # Extraction method
        self.table_method_combo = QComboBox()
        set_control_height(self.table_method_combo)
        self.table_method_combo.addItems(["Camelot", "PDFPlumber", "Transformer"])
        self.table_method_combo.setToolTip("Table extraction method. Camelot (ML-based on PDF structure), PDFPlumber (heuristics on text/lines), Transformer (DL-based for scanned documents).")
        method_label = QLabel("Extraction Method:")
        method_label.setToolTip(self.table_method_combo.toolTip())
        basic_form.addRow(method_label, self.table_method_combo)

        # Flavor (depends on method)
        self.table_flavor_combo = QComboBox()
        set_control_height(self.table_flavor_combo)
        self.table_flavor_combo.setToolTip("Extraction flavor. Camelot: lattice/stream/auto. PDFPlumber: lines/text/auto. Transformer has no flavor.")
        self.table_flavor_label = QLabel("Flavor:")
        self.table_flavor_label.setToolTip(self.table_flavor_combo.toolTip())
        basic_form.addRow(self.table_flavor_label, self.table_flavor_combo)
        
        # Add hint label for flavor
        self.flavor_hint_label = QLabel("")
        self.flavor_hint_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        self.flavor_hint_label.setWordWrap(True)
        basic_form.addRow("", self.flavor_hint_label)

        # Output path
        self.btn_output = QPushButton("Select Output Folder")
        set_control_height(self.btn_output, 24)
        self.lbl_output = QLabel("")
        set_control_height(self.lbl_output, 24)
        self.lbl_output.setStyleSheet("border: 1px solid #aaa; border-radius: 4px; padding: 2px 6px; background: #fafbfc;")
        self.lbl_output.setToolTip("Output folder for extracted data and images")
        output_label = QLabel("Output Path:")
        output_label.setToolTip("Select the folder where extracted tables and images will be saved")
        basic_form.addRow(output_label, self.lbl_output)
        basic_form.addRow("", self.btn_output)

        # Export format
        self.cmb_export = QComboBox()
        set_control_height(self.cmb_export)
        self.cmb_export.addItems(['CSV', 'JSON'])
        self.cmb_export.setToolTip("Output format for extracted table data. CSV is most compatible, JSON preserves structure, Excel includes formatting.")
        export_label = QLabel("Export Format:")
        export_label.setToolTip(self.cmb_export.toolTip())
        basic_form.addRow(export_label, self.cmb_export)

        # Page selection
        pages_layout = QHBoxLayout()
        self.pages_all_radio = QRadioButton("All Pages")
        self.pages_all_radio.setChecked(True)
        self.pages_custom_radio = QRadioButton("Custom Range:")
        self.pages_input = QLineEdit()
        set_control_height(self.pages_input)
        self.pages_input.setPlaceholderText("e.g. 1-3,5,7")
        self.pages_input.setEnabled(False)
        self.pages_input.setToolTip("Specify page range to process. Examples: '1-3' for pages 1-3, '1,3,5' for specific pages, '2-' for page 2 onwards")
        pages_layout.addWidget(self.pages_all_radio)
        pages_layout.addWidget(self.pages_custom_radio)
        pages_layout.addWidget(self.pages_input)
        pages_group = QGroupBox("Pages to Process")
        pages_group.setLayout(pages_layout)
        pages_group.setToolTip("Select which pages to process from the PDF file")
        basic_form.addRow(pages_group)

        # Save images option
        """ self.save_images_checkbox = QCheckBox("Save PDF Images Separately")
        self.save_images_checkbox.setToolTip("Extract and save images from PDF to output/images/filename/ folder. Images will be named by their original name or as filename_page_index_image_index")
        save_images_label = QLabel("Save Images:")
        save_images_label.setToolTip(self.save_images_checkbox.toolTip())
        basic_form.addRow(save_images_label, self.save_images_checkbox) """

        basic_tab.setLayout(basic_form)
        self.tabs.addTab(basic_tab, "Basic Parameters")

        # ========== Advanced Parameters Tab ==========
        adv_tab = QWidget()
        adv_form = QFormLayout()
        adv_form.setVerticalSpacing(12)


        # Table score threshold
        self.table_score_spin = QDoubleSpinBox()
        set_control_height(self.table_score_spin)
        self.table_score_spin.setRange(0.0, 1.0)
        self.table_score_spin.setSingleStep(0.05)
        # self.table_score_spin.setValue(0.6)
        self.table_score_spin.setToolTip("Minimum confidence score for table detection with Camelot and pdfplumber. Higher values are more strict. Recommended: 0.5-0.8")
        score_label = QLabel("Table Score Threshold:")
        score_label.setToolTip(self.table_score_spin.toolTip())
        adv_form.addRow(score_label, self.table_score_spin)

        # Table deduplication IoU threshold
        self.table_iou_spin = QDoubleSpinBox()
        set_control_height(self.table_iou_spin)
        self.table_iou_spin.setRange(0.0, 1.0)
        self.table_iou_spin.setSingleStep(0.05)
        # self.table_iou_spin.setValue(0.3)
        self.table_iou_spin.setToolTip("IoU threshold for table deduplication. Lower values remove more overlapping tables. Recommended: 0.2-0.5")
        iou_label = QLabel("Table Deduplication IoU:")
        iou_label.setToolTip(self.table_iou_spin.toolTip())
        adv_form.addRow(iou_label, self.table_iou_spin)

        # Transformer detection threshold
        self.transformer_detection_spin = QDoubleSpinBox()
        set_control_height(self.transformer_detection_spin)
        self.transformer_detection_spin.setRange(0.0, 1.0)
        self.transformer_detection_spin.setSingleStep(0.05)
        # self.transformer_detection_spin.setValue(0.5)
        self.transformer_detection_spin.setToolTip("Confidence threshold for transformer-based table detection. Higher is more strict. Recommended: 0.4-0.7")
        transformer_det_label = QLabel("Transformer Detection Threshold:")
        transformer_det_label.setToolTip(self.transformer_detection_spin.toolTip())
        adv_form.addRow(transformer_det_label, self.transformer_detection_spin)

        # Transformer structure recognition threshold
        self.transformer_structure_spin = QDoubleSpinBox()
        set_control_height(self.transformer_structure_spin)
        self.transformer_structure_spin.setRange(0.0, 1.0)
        self.transformer_structure_spin.setSingleStep(0.05)
        # self.transformer_structure_spin.setValue(0.5)
        self.transformer_structure_spin.setToolTip("Confidence threshold for transformer-based structure recognition. Higher is more strict. Recommended: 0.4-0.7")
        transformer_struct_label = QLabel("Transformer Structure Threshold:")
        transformer_struct_label.setToolTip(self.transformer_structure_spin.toolTip())
        adv_form.addRow(transformer_struct_label, self.transformer_structure_spin)

        # Tesseract OCR threshold
        self.tesseract_threshold_spin = QDoubleSpinBox()
        set_control_height(self.tesseract_threshold_spin)
        self.tesseract_threshold_spin.setRange(0.0, 1.0)
        self.tesseract_threshold_spin.setSingleStep(0.05)
        # self.tesseract_threshold_spin.setValue(0.5)
        self.tesseract_threshold_spin.setToolTip("Confidence threshold for Tesseract OCR text recognition. Higher is more strict. Recommended: 0.4-0.8")
        tesseract_label = QLabel("Tesseract OCR Threshold:")
        tesseract_label.setToolTip(self.tesseract_threshold_spin.toolTip())
        adv_form.addRow(tesseract_label, self.tesseract_threshold_spin)

        # Transformer structure preprocessing
        self.transformer_preprocess_combo = QComboBox()
        set_control_height(self.transformer_preprocess_combo)
        self.transformer_preprocess_combo.addItems(["Yes", "No"])
        # self.transformer_preprocess_combo.setCurrentIndex(0)
        self.transformer_preprocess_combo.setToolTip("Enable preprocessing for transformer structure recognition. May improve accuracy for complex tables. Recommended: Yes")
        preprocess_label = QLabel("Transformer Structure Preprocessing:")
        preprocess_label.setToolTip(self.transformer_preprocess_combo.toolTip())
        adv_form.addRow(preprocess_label, self.transformer_preprocess_combo)

        adv_tab.setLayout(adv_form)
        self.tabs.addTab(adv_tab, "Advanced Parameters")

        # --- Button area ---
        btn_layout = QHBoxLayout()
        self.btn_process = QPushButton("Start Processing")
        set_control_height(self.btn_process, 36)
        self.btn_restore = QPushButton("Restore Defaults")
        set_control_height(self.btn_restore, 36)
        self.btn_process.setMinimumWidth(200)
        self.btn_restore.setMinimumWidth(200)
        self.btn_process.setStyleSheet("font-weight: bold; padding: 10px 0;")
        self.btn_restore.setStyleSheet("font-weight: bold; padding: 10px 0;")
        btn_layout.addWidget(self.btn_process, 1)
        btn_layout.addWidget(self.btn_restore, 1)
        btn_layout.addStretch()

        # 阻塞所有会触发_update_params的控件的信号
        widgets_to_block = [
            self.cmb_export, self.pages_all_radio, self.pages_custom_radio,
            #self.save_images_checkbox, self.table_method_combo, self.table_flavor_combo,
            self.table_score_spin, self.table_iou_spin, self.transformer_detection_spin,
            self.transformer_structure_spin, self.tesseract_threshold_spin,
            self.transformer_preprocess_combo, self.pages_input
        ]
        
        # 保存原始信号状态
        original_states = [widget.signalsBlocked() for widget in widgets_to_block]
        
        # 阻塞信号
        for widget in widgets_to_block:
            widget.blockSignals(True)
        # 使用self.params设置控件初始值
        self._set_ui_values_from_params()
        
        # 恢复信号状态
        for i, widget in enumerate(widgets_to_block):
            widget.blockSignals(original_states[i])
        
        # 连接信号
        self._connect_signals()
        
        # Assemble main layout
        main_layout.addWidget(self.tabs)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        # 手动调用一次_update_params以确保参数同步
        self._update_params()
    
    def _update_flavor_options(self, block_signals=False):
        """Update flavor options based on method without triggering update_params"""
        if block_signals:
            self.table_flavor_combo.blockSignals(True)
        
        try:
            method = self.table_method_combo.currentText().lower()
            self.table_flavor_combo.clear()
            
            if method == 'camelot':
                self.table_flavor_combo.addItems(['Lattice', 'Stream', 'Auto'])
                self.table_flavor_combo.setCurrentIndex(0)
                self.table_flavor_combo.setVisible(True)
                self.table_flavor_label.setVisible(True)
                self.flavor_hint_label.setText("Auto模式将根据页面特征自动选择最佳flavor")
            elif method == 'pdfplumber':
                self.table_flavor_combo.addItems(['Lines', 'Text', 'Auto'])
                self.table_flavor_combo.setCurrentIndex(0)
                self.table_flavor_combo.setVisible(True)
                self.table_flavor_label.setVisible(True)
                self.flavor_hint_label.setText("Auto模式将根据页面特征自动选择最佳flavor")
            elif method == 'transformer':
                self.table_flavor_combo.setVisible(False)
                self.table_flavor_label.setVisible(False)
                self.flavor_hint_label.setText("Transformer仅支持本地模型，适用于扫描文档")
            else:
                self.table_flavor_combo.setVisible(False)
                self.table_flavor_label.setVisible(False)
                self.flavor_hint_label.setText("")
        
        finally:
            if block_signals:
                self.table_flavor_combo.blockSignals(False)

    def _on_method_changed(self):
        """Update flavor options based on method and update params."""
        self._update_flavor_options()
        self._update_params()

    def _update_params(self):
        """Update parameters and auto-save to config"""
        # Basic parameters
        if self.pages_all_radio.isChecked():
            pages_value = 'all'
        else:
            pages_value = self.pages_input.text().strip()
        
        params_update = {
            'output_path': self.lbl_output.text(),
            'export_format': self.cmb_export.currentText().lower(),
            'pages': pages_value,
            #'save_images': self.save_images_checkbox.isChecked(),
            'table_method': self.table_method_combo.currentText().lower(),
            'table_flavor': (self.table_flavor_combo.currentText().lower() if self.table_flavor_combo.isVisible() else None),
            'table_score_threshold': self.table_score_spin.value(),
            'table_iou_threshold': self.table_iou_spin.value(),
            'transformer_detection_threshold': self.transformer_detection_spin.value(),
            'transformer_structure_threshold': self.transformer_structure_spin.value(),
            'tesseract_threshold': self.tesseract_threshold_spin.value(),
            'transformer_preprocess': self.transformer_preprocess_combo.currentText() == "Yes"
        }
        self.params.update(params_update)
        self.paramsChanged.emit(self.params.copy())

        # Auto-save to config
        self._save_params_to_config()

    def _save_params_to_config(self):
        """Save current parameters to config.json"""
        try:
            # Update config with current parameters
            self.config.set('ui', self.params.copy())
            self.config.save()
        except Exception as e:
            # 调试信息已移除
            pass

    def get_params(self):
        """Get current parameters for processing"""
        if not self.params.get('output_path'):
            raise ValueError("Output path must be selected")
        
        # Return a copy of current parameters
        return self.params.copy()

    def _select_output(self):
        """Select output folder for extracted data"""
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.lbl_output.setText(path)
            self.params['output_path'] = path
            # Set logger output path
            logger = AppLogger.get_logger()
            logger.set_output_path(path)
            self._update_params()

    def load_params_from_file(self):
        """Load parameters from config file"""
        self.config.load()
        ui_cfg = self.config.get('ui', {})
        
        # 更新内部参数（此时self.params已存在）
        self.params.update(ui_cfg)

    def _connect_signals(self):
        """Connect all signals to their respective slots"""
        self.btn_output.clicked.connect(self._select_output)
        self.cmb_export.currentIndexChanged.connect(self._update_params)
        self.pages_all_radio.toggled.connect(lambda checked: self.pages_input.setEnabled(not checked))
        self.pages_all_radio.toggled.connect(self._update_params)
        self.pages_custom_radio.toggled.connect(self._update_params)
        self.pages_input.textChanged.connect(self._update_params)
        #self.save_images_checkbox.toggled.connect(self._update_params)
        self.table_method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.table_flavor_combo.currentIndexChanged.connect(self._update_params)
        self.table_score_spin.valueChanged.connect(self._update_params)
        self.table_iou_spin.valueChanged.connect(self._update_params)
        self.transformer_detection_spin.valueChanged.connect(self._update_params)
        self.transformer_structure_spin.valueChanged.connect(self._update_params)
        self.tesseract_threshold_spin.valueChanged.connect(self._update_params)
        self.transformer_preprocess_combo.currentIndexChanged.connect(self._update_params)
        self.btn_process.clicked.connect(self._start_processing)
        self.btn_restore.clicked.connect(self._restore_defaults)

    
    def _set_ui_values_from_params(self):
        """Set UI control values from self.params"""
        # 输出路径
        output_path = self.params.get('output_path', '')
        self.lbl_output.setText(output_path)
        
        # 导出格式
        export_format = self.params.get('export_format', 'csv').upper()
        index = self.cmb_export.findText(export_format, Qt.MatchFixedString)
        if index >= 0:
            self.cmb_export.setCurrentIndex(index)
        else:
            self.cmb_export.setCurrentIndex(0)
        
        # 页面选择
        pages_value = self.params.get('pages', 'all')
        if pages_value == 'all':
            self.pages_all_radio.setChecked(True)
            self.pages_input.setEnabled(False)
        else:
            self.pages_custom_radio.setChecked(True)
            self.pages_input.setEnabled(True)
            self.pages_input.setText(pages_value)
        
        # 保存图片选项
        #save_images = self.params.get('save_images', False)
        #self.save_images_checkbox.setChecked(save_images)
        
        # 提取方法
        method = self.params.get('table_method', 'camelot').title()
        index = self.table_method_combo.findText(method, Qt.MatchFixedString)
        if index >= 0:
            self.table_method_combo.setCurrentIndex(index)
        else:
            self.table_method_combo.setCurrentIndex(0)
        
        # 更新风格选项
        self._update_flavor_options(block_signals=True)
        
        # 提取风格
        flavor = self.params.get('table_flavor')
        if flavor and self.table_flavor_combo.isVisible():
            for i in range(self.table_flavor_combo.count()):
                if self.table_flavor_combo.itemText(i).lower() == str(flavor).lower():
                    self.table_flavor_combo.setCurrentIndex(i)
                    break
        
        # 表格分数阈值
        table_score = self.params.get('table_score_threshold', 0.6)
        self.table_score_spin.setValue(table_score)
        
        # 表格IoU阈值
        table_iou = self.params.get('table_iou_threshold', 0.3)
        self.table_iou_spin.setValue(table_iou)
        
        # Transformer检测阈值
        transformer_detection = self.params.get('transformer_detection_threshold', 0.5)
        self.transformer_detection_spin.setValue(transformer_detection)
        
        # Transformer结构阈值
        transformer_structure = self.params.get('transformer_structure_threshold', 0.5)
        self.transformer_structure_spin.setValue(transformer_structure)
        
        # Tesseract阈值
        tesseract_threshold = self.params.get('tesseract_threshold', 0.5)
        self.tesseract_threshold_spin.setValue(tesseract_threshold)
        
        # Transformer预处理
        transformer_preprocess = self.params.get('transformer_preprocess', True)
        self.transformer_preprocess_combo.setCurrentIndex(0 if transformer_preprocess else 1)


    def closeEvent(self, event):
        """Save parameters when closing"""
        self._save_params_to_config()
        super().closeEvent(event)

    def showEvent(self, event):
        """Load parameters when showing (fallback)"""
        # 只在控件不存在时才重新加载（作为fallback）
        if not hasattr(self, 'lbl_output'):
            self.load_params_from_file()
        super().showEvent(event)

    def _start_processing(self):
        """Start processing with current parameters"""
        try:
            params = self.get_params()
            if not params.get('output_path'):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Missing Output Folder", "Please select an output folder before processing.")
                return
            
            # Validate file list
            main_window = self.window()
            file_panel = getattr(main_window, 'file_panel', None)
            if file_panel and hasattr(file_panel, '_get_file_list'):
                files = file_panel._get_file_list()
                if not files:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "No Files", "Please add at least one file before processing.")
                    return
            
            # Save parameters and start processing
            self._save_params_to_config()
            self.process_started.emit(params)
        except ValueError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Parameter Error", str(e))
            return

    def _restore_defaults(self):
        """Restore all parameters to default values"""
        # Basic parameters
        self.lbl_output.setText("")
        self.cmb_export.setCurrentIndex(0)  # CSV
        self.pages_all_radio.setChecked(True)
        self.pages_input.setText("")
        #self.save_images_checkbox.setChecked(False)
        
        # Advanced parameters
        self.table_method_combo.setCurrentIndex(0)  # Camelot
        self.table_score_spin.setValue(0.6)
        self.table_iou_spin.setValue(0.3)
        self.transformer_detection_spin.setValue(0.5)
        self.transformer_structure_spin.setValue(0.5)
        self.tesseract_threshold_spin.setValue(0.5)
        self.transformer_preprocess_combo.setCurrentIndex(0)  # Yes
        
        # Update parameters
        self._update_params()
