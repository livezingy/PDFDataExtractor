# gui/panels/params_panel.py
from PySide6.QtWidgets import (QWidget, QFormLayout, QComboBox, QLineEdit, QPushButton, QLabel, QFileDialog,
                              QRadioButton, QButtonGroup, QHBoxLayout, QVBoxLayout,
                              QGroupBox, QSpinBox, QDoubleSpinBox, QTabWidget)
from PySide6.QtCore import Signal, Qt, QEvent, QPoint
from core.utils.logger import AppLogger
import os
import json


PARAMS_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".pdfdataextractor_params.json")


class ParamsPanel(QWidget):
    paramsChanged = Signal(dict)
    process_started = Signal(dict)  # Add process_started signal
    
    def __init__(self):
        super().__init__()  # Call base class initialization first
        
        self.params = {
            'export_format': 'csv',
            'output_path': '',
            'pages': 'all',  # Default to all pages
            'flavor': 'hybrid',
            'current_filepath': ''        }
        self._init_ui()
        
        # Ensure panel is visible
        self.setMinimumHeight(250)
        self.setMaximumHeight(400)
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

        # Camelot Parameters GroupBox (composite box: accuracy threshold, mode dropdown, process_background checkbox)
        camelot_group = QGroupBox("Camelot Parameters")
        
        camelot_inner_form = QFormLayout()
        camelot_inner_form.setVerticalSpacing(16)  # Double row spacing
        # Camelot accuracy threshold
        self.camelot_threshold_spin = QDoubleSpinBox()
        set_control_height(self.camelot_threshold_spin)
        self.camelot_threshold_spin.setMinimumWidth(120)
        self.camelot_threshold_spin.setMaximumWidth(180)
        self.camelot_threshold_spin.setRange(0.0, 1.0)
        self.camelot_threshold_spin.setSingleStep(0.05)
        self.camelot_threshold_spin.setValue(0.6)
        self.camelot_threshold_spin.setToolTip("Minimum confidence threshold for Camelot table detection. Higher is stricter. Recommended: 0.5-0.8")
        camelot_accuracy_label = QLabel("Camelot Accuracy Threshold:")
        camelot_accuracy_label.setToolTip(self.camelot_threshold_spin.toolTip())
        camelot_accuracy_label.setMinimumWidth(170)
        camelot_accuracy_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        camelot_inner_form.addRow(camelot_accuracy_label, self.camelot_threshold_spin)
        # Camelot mode dropdown
        self.camelot_mode_combo = QComboBox()
        set_control_height(self.camelot_mode_combo)
        self.camelot_mode_combo.setMinimumWidth(120)
        self.camelot_mode_combo.setMaximumWidth(180)
        self.camelot_mode_combo.addItems(["Hybrid", "Lattice", "Stream", "Network"])
        self.camelot_mode_combo.setCurrentIndex(0)
        self.camelot_mode_combo.setToolTip("Camelot table recognition mode. Hybrid auto-selects best, Lattice for ruled tables, Stream for borderless, Network for deep learning.")
        camelot_mode_label = QLabel("Camelot Mode:")
        camelot_mode_label.setToolTip(self.camelot_mode_combo.toolTip())
        camelot_mode_label.setMinimumWidth(170)
        camelot_mode_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        camelot_inner_form.addRow(camelot_mode_label, self.camelot_mode_combo)
        # process_background parameter
        self.camelot_process_bg_checkbox = QComboBox()
        set_control_height(self.camelot_process_bg_checkbox)
        self.camelot_process_bg_checkbox.setMinimumWidth(120)
        self.camelot_process_bg_checkbox.setMaximumWidth(180)
        self.camelot_process_bg_checkbox.addItems(["Yes", "No"])
        self.camelot_process_bg_checkbox.setToolTip("Available in Lattice mode. Enabling may improve detection for complex tables.")
        camelot_process_bg_label = QLabel("Process Background:")
        camelot_process_bg_label.setToolTip(self.camelot_process_bg_checkbox.toolTip())
        camelot_process_bg_label.setMinimumWidth(170)
        camelot_process_bg_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        camelot_inner_form.addRow(camelot_process_bg_label, self.camelot_process_bg_checkbox)
        camelot_group.setLayout(camelot_inner_form)
        basic_form.setVerticalSpacing(16)  # Double row spacing for basic tab
        basic_form.addRow(camelot_group)

        # Page selection
        pages_layout = QHBoxLayout()
        self.pages_all_radio = QRadioButton("All")
        self.pages_all_radio.setChecked(True)
        self.pages_custom_radio = QRadioButton("Custom:")
        self.pages_input = QLineEdit()
        set_control_height(self.pages_input)
        self.pages_input.setPlaceholderText("e.g. 1-3,5,7")
        self.pages_input.setEnabled(False)
        pages_layout.addWidget(self.pages_all_radio)
        pages_layout.addWidget(self.pages_custom_radio)
        pages_layout.addWidget(self.pages_input)
        pages_group = QGroupBox("Pages to Process:")
        pages_group.setLayout(pages_layout)
        basic_form.addRow(pages_group)
        self.pages_all_radio.toggled.connect(lambda checked: self.pages_input.setEnabled(not checked))
        self.pages_all_radio.toggled.connect(self._update_params)
        self.pages_custom_radio.toggled.connect(self._update_params)
        self.pages_input.textChanged.connect(self._update_params)

        # Export format
        self.cmb_export = QComboBox()
        set_control_height(self.cmb_export)
        self.cmb_export.addItems(['CSV', 'JSON'])
        basic_form.addRow("Export Format:", self.cmb_export)

        # Output path
        self.btn_output = QPushButton("Select Folder")
        set_control_height(self.btn_output, 24)
        self.lbl_output = QLabel("")
        set_control_height(self.lbl_output, 24)
        self.lbl_output.setStyleSheet("border: 1px solid #aaa; border-radius: 4px; padding: 2px 6px; background: #fafbfc;")
        basic_form.addRow(self.btn_output, self.lbl_output)
        self.params['output_path'] = self.lbl_output.text()

        # Output path info
        self.lbl_output_info = QLabel(
            "Output folder will contain the following subfolders:\n"
            "- data: Extracted table data\n"
            "- debug: Debug information\n"
            "- preview: Preview images"
        )
        self.lbl_output_info.setStyleSheet("color: #666; font-size: 10pt;")
        self.lbl_output_info.setWindowFlags(Qt.ToolTip)
        self.lbl_output_info.setVisible(False)
        self.btn_output.installEventFilter(self)
        self.lbl_output.installEventFilter(self)

        basic_tab.setLayout(basic_form)
        self.tabs.addTab(basic_tab, "Basic Parameters")

        # ========== Advanced Parameters Tab ==========
        adv_tab = QWidget()
        adv_form = QFormLayout()
        adv_form.setVerticalSpacing(16)  # Double row spacing
        # Table detection threshold
        self.detection_threshold_spin = QDoubleSpinBox()
        set_control_height(self.detection_threshold_spin)
        self.detection_threshold_spin.setRange(0.0, 1.0)
        self.detection_threshold_spin.setSingleStep(0.05)
        self.detection_threshold_spin.setValue(0.5)
        self.detection_threshold_spin.setToolTip("Confidence threshold for table detection model. Higher is stricter. Recommended: 0.4-0.7")
        detection_label = QLabel("Table detection threshold:")
        detection_label.setToolTip(self.detection_threshold_spin.toolTip())
        adv_form.addRow(detection_label, self.detection_threshold_spin)
        # Table structure recognition threshold
        self.structure_threshold_spin = QDoubleSpinBox()
        set_control_height(self.structure_threshold_spin)
        self.structure_threshold_spin.setRange(0.0, 1.0)
        self.structure_threshold_spin.setSingleStep(0.05)
        self.structure_threshold_spin.setValue(0.5)
        self.structure_threshold_spin.setToolTip("Confidence threshold for structure recognition model. Higher is stricter. Recommended: 0.4-0.7")
        structure_label = QLabel("Table structure recognition threshold:")
        structure_label.setToolTip(self.structure_threshold_spin.toolTip())
        adv_form.addRow(structure_label, self.structure_threshold_spin)
        # Structure border width
        self.structure_border_spin = QSpinBox()
        set_control_height(self.structure_border_spin)
        self.structure_border_spin.setRange(0, 100)
        self.structure_border_spin.setValue(20)
        self.structure_border_spin.setToolTip("Structure border width (pixels). Adjust as needed for different tables.")
        border_label = QLabel("Structure Border Width (px):")
        border_label.setToolTip(self.structure_border_spin.toolTip())
        adv_form.addRow(border_label, self.structure_border_spin)
        # Structure preprocessing
        self.structure_preprocess_checkbox = QComboBox()
        set_control_height(self.structure_preprocess_checkbox)
        self.structure_preprocess_checkbox.addItems(["Yes", "No"])
        self.structure_preprocess_checkbox.setCurrentIndex(0)
        self.structure_preprocess_checkbox.setToolTip("Whether to preprocess images to improve structure recognition.")
        preprocess_label = QLabel("Structure Preprocessing:")
        preprocess_label.setToolTip(self.structure_preprocess_checkbox.toolTip())
        adv_form.addRow(preprocess_label, self.structure_preprocess_checkbox)
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

        # Signal connections
        self.btn_output.clicked.connect(self._select_output)
        self.cmb_export.currentIndexChanged.connect(self._update_params)
        self.camelot_mode_combo.currentIndexChanged.connect(self._update_params)
        self.camelot_process_bg_checkbox.currentIndexChanged.connect(self._update_params)
        self.camelot_threshold_spin.valueChanged.connect(self._update_params)
        self.detection_threshold_spin.valueChanged.connect(self._update_params)
        self.structure_threshold_spin.valueChanged.connect(self._update_params)
        self.structure_border_spin.valueChanged.connect(self._update_params)
        self.structure_preprocess_checkbox.currentIndexChanged.connect(self._update_params)
        self.btn_process.clicked.connect(self._start_processing)
        self.btn_restore.clicked.connect(self._restore_defaults)

        # Assemble main layout
        main_layout.addWidget(self.tabs)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj in [self.btn_output, self.lbl_output]:
            if event.type() == QEvent.Enter:
                # Show floating tooltip near the button
                global_pos = obj.mapToGlobal(obj.rect().bottomLeft())
                self.lbl_output_info.move(global_pos + QPoint(0, 8))
                self.lbl_output_info.setVisible(True)
            elif event.type() == QEvent.Leave:
                self.lbl_output_info.setVisible(False)
            elif event.type() == QEvent.MouseButtonPress:
                self.lbl_output_info.setVisible(False)
        return super().eventFilter(obj, event)

    def _select_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.lbl_output.setText(path)
            self.params['output_path'] = path
            # Set logger output path
            logger = AppLogger.get_logger()
            logger.set_output_path(path)
            self.paramsChanged.emit(self.params)

    def save_params_to_file(self, params=None):
        if params is None:
            params = self.get_params()
        try:
            with open(PARAMS_CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger = AppLogger.get_logger()
            logger.error(f"Failed to save parameters: {e}")

    def load_params_from_file(self):
        if not os.path.exists(PARAMS_CONFIG_PATH):
            return
        try:
            with open(PARAMS_CONFIG_PATH, 'r', encoding='utf-8') as f:
                params = json.load(f)
            self.apply_params(params)
        except Exception as e:
            logger = AppLogger.get_logger()
            logger.error(f"Failed to load parameters from {PARAMS_CONFIG_PATH}: {e}")

    def apply_params(self, params: dict):
        # Set all UI controls from params dict
        self.camelot_mode_combo.setCurrentText(params.get('camelot_mode', 'hybrid').capitalize())
        self.camelot_threshold_spin.setValue(params.get('camelot_accuracy_threshold', 0.6))
        self.detection_threshold_spin.setValue(params.get('detection_threshold', 0.5))
        self.structure_threshold_spin.setValue(params.get('structure_threshold', 0.5))
        self.structure_border_spin.setValue(params.get('structure_border_width', 20))
        self.structure_preprocess_checkbox.setCurrentIndex(0 if params.get('structure_preprocess', True) else 1)
        self.camelot_process_bg_checkbox.setCurrentIndex(0 if params.get('process_background', False) else 1)
        self.cmb_export.setCurrentText(params.get('export_format', 'csv').upper())
        self.lbl_output.setText(params.get('output_path', ''))
        if params.get('pages', 'all') == 'all':
            self.pages_all_radio.setChecked(True)
            self.pages_input.setText("")
        else:
            self.pages_custom_radio.setChecked(True)
            self.pages_input.setText(params.get('pages', ''))
        self.params.update(params)
        self._update_params()

    def closeEvent(self, event):
        self.save_params_to_file()
        super().closeEvent(event)

    def showEvent(self, event):
        self.load_params_from_file()
        super().showEvent(event)

    def _start_processing(self):
        try:
            params = self.get_params()
            if not params.get('output_path'):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Missing Output Folder", "Please select an output folder before processing.")
                return
            # 新增：校验文件列表
            main_window = self.window()  # 更通用的方式获取主窗口
            file_panel = getattr(main_window, 'file_panel', None)
            if file_panel and hasattr(file_panel, '_get_file_list'):
                files = file_panel._get_file_list()
                if not files:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "No Files", "Please add at least one file before processing.")
                    return
            self.save_params_to_file(params)
            self.process_started.emit(params)
        except ValueError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Parameter Error", str(e))
            return
   

    def _update_params(self):
        # Update camelot_mode parameter
        self.params['flavor'] = self.camelot_mode_combo.currentText().lower()
        # Update pages parameter
        if self.pages_all_radio.isChecked():
            self.params['pages'] = 'all'
        else:
            self.params['pages'] = self.pages_input.text().strip()
        self.params.update({
            'export_format': self.cmb_export.currentText().lower(),
            'camelot_accuracy_threshold': self.camelot_threshold_spin.value(),
            'detection_threshold': self.detection_threshold_spin.value(),
            'structure_threshold': self.structure_threshold_spin.value(),
            'structure_border_width': self.structure_border_spin.value(),
            'structure_preprocess': self.structure_preprocess_checkbox.currentText() == "Yes",
            'process_background': self.camelot_process_bg_checkbox.currentText() == "Yes"
        })
        self.paramsChanged.emit(self.params.copy())

    def _on_params_changed(self):
        params = self.get_params()
        self.paramsChanged.emit(params)

    def get_params(self):
        if not self.params['output_path']:
            raise ValueError("Output path must be selected")
        # pages parameter handling
        if self.pages_all_radio.isChecked():
            pages_value = 'all'
        else:
            pages_value = self.pages_input.text().strip()
        return {
            'flavor': self.camelot_mode_combo.currentText().lower(),
            'export_format': self.cmb_export.currentText().lower(),
            'output_path': self.lbl_output.text(),
            'pages': pages_value,
            'current_filepath': self.params.get('current_filepath', ''),
            'camelot_accuracy_threshold': self.camelot_threshold_spin.value(),
            'detection_threshold': self.detection_threshold_spin.value(),
            'structure_threshold': self.structure_threshold_spin.value(),
            'structure_border_width': self.structure_border_spin.value(),
            'structure_preprocess': self.structure_preprocess_checkbox.currentText() == "Yes",
            'process_background': self.camelot_process_bg_checkbox.currentText() == "Yes"
        }

    def _restore_defaults(self):
        # Restore all parameters to default values
        self.camelot_threshold_spin.setValue(0.6)
        self.detection_threshold_spin.setValue(0.5)
        self.structure_threshold_spin.setValue(0.5)
        self.structure_border_spin.setValue(20)
        self.structure_preprocess_checkbox.setCurrentIndex(0)
        self.cmb_export.setCurrentIndex(0)
        self.pages_all_radio.setChecked(True)
        self.pages_input.setText("")
        self.camelot_mode_combo.setCurrentIndex(0)
        self.camelot_process_bg_checkbox.setCurrentIndex(1)  # Default to No
        self.lbl_output.setText("")
        self.params['output_path'] = ""
        self._update_params()
