# gui/panels/params_panel.py
from PySide6.QtWidgets import (QWidget, QFormLayout, QComboBox, 
                              QLineEdit, QPushButton, QLabel, QFileDialog,
                              QRadioButton, QButtonGroup, QHBoxLayout, QVBoxLayout,
                              QGroupBox, QSpinBox, QDoubleSpinBox)
from PySide6.QtCore import Signal, Qt, QEvent, QPoint
from core.utils.logger import AppLogger


class ParamsPanel(QWidget):
    paramsChanged = Signal(dict)
    process_started = Signal(dict)  # Add process_started signal
    
    def __init__(self):
        super().__init__()  # Call base class initialization first
        
        self.params = {
            'export_format': 'csv',
            'output_path': '',
            'pages': 'all',  # Default to all pages
            'camelot_mode': 'hybrid',
            'current_filepath': ''        }
        self._init_ui()
        
        # Ensure panel is visible
        self.setMinimumHeight(250)
        self.setMaximumHeight(400)
        self.setVisible(True)

    def _init_ui(self):
        layout = QFormLayout()
        
        # Set panel style
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f4f8;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox, QLineEdit, QPushButton {
                min-height: 25px;
                padding: 3px;
            }
            QLabel {
                font-weight: bold;
            }
        """)
        
        # Add Camelot processing mode selection
        self.camelotMode_group = QButtonGroup()
        self.lattice_radio = QRadioButton("Lattice")
        self.stream_radio = QRadioButton("Stream")
        self.network_radio = QRadioButton("Network")
        self.hybrid_radio = QRadioButton("Hybrid")
        self.camelotMode_group.addButton(self.lattice_radio)
        self.camelotMode_group.addButton(self.stream_radio)
        self.camelotMode_group.addButton(self.network_radio)
        self.camelotMode_group.addButton(self.hybrid_radio)
        self.hybrid_radio.setChecked(True)  # Default to Hybrid mode
        camelot_layout = QHBoxLayout()
        camelot_layout.addWidget(self.lattice_radio)
        camelot_layout.addWidget(self.stream_radio)
        camelot_layout.addWidget(self.network_radio)
        camelot_layout.addWidget(self.hybrid_radio)
        camelot_group = QGroupBox("Camelot Mode: ")
        camelot_group.setLayout(camelot_layout)
        layout.addRow(camelot_group)

        # Pages selection area
        pages_layout = QHBoxLayout()
        self.pages_all_radio = QRadioButton("All")
        self.pages_all_radio.setChecked(True)
        self.pages_custom_radio = QRadioButton("Custom:")
        self.pages_input = QLineEdit()
        self.pages_input.setPlaceholderText("e.g. 1-3,5,7")
        self.pages_input.setEnabled(False)
        pages_layout.addWidget(self.pages_all_radio)
        pages_layout.addWidget(self.pages_custom_radio)
        pages_layout.addWidget(self.pages_input)
        pages_group = QGroupBox("Pages to Process:")
        pages_group.setLayout(pages_layout)
        layout.addRow(pages_group)
        # Connect signals for pages input
        self.pages_all_radio.toggled.connect(lambda checked: self.pages_input.setEnabled(not checked))
        self.pages_all_radio.toggled.connect(self._update_params)
        self.pages_custom_radio.toggled.connect(self._update_params)
        self.pages_input.textChanged.connect(self._update_params)

        # Export Format
        self.cmb_export = QComboBox()
        self.cmb_export.addItems(['CSV', 'JSON'])
        layout.addRow("Export Format:", self.cmb_export)
        
        # Output Path
        self.btn_output = QPushButton("Select Folder")
        self.lbl_output = QLabel(r"")
        layout.addRow(self.btn_output, self.lbl_output)
        self.params['output_path'] = self.lbl_output.text()

        # Add output path information
        self.lbl_output_info = QLabel(
            "Output folder will contain the following subfolders:\n"
            "- data: Extracted table data\n"
            "- debug: Debug information\n"
            "- preview: Preview images"
        )
        self.lbl_output_info.setStyleSheet("color: #666; font-size: 10pt;")
        self.lbl_output_info.setWindowFlags(Qt.ToolTip)
        self.lbl_output_info.setVisible(False)

        #hover event
        self.btn_output.installEventFilter(self)
        self.lbl_output.installEventFilter(self)

        # Add Process button
        self.btn_process = QPushButton("Start Processing")
        self.btn_process.clicked.connect(self._start_processing)
        layout.addRow(self.btn_process)
        
        # Signal Connections
        self.btn_output.clicked.connect(self._select_output)
        self.cmb_export.currentIndexChanged.connect(self._update_params)
        self.camelotMode_group.buttonClicked.connect(self._update_params)        
        
        self.setLayout(layout)

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

    def _start_processing(self):
        """Start processing with current parameters"""
        try:
            params = self.get_params()
            # Check output path
            if not params.get('output_path'):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Missing Output Folder", "Please select an output folder before processing.")
                return
            self.process_started.emit(params)
        except ValueError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Parameter Error", str(e))
            return
   

    def _update_params(self):
        # Update camelot_mode parameter
        checked_button = self.camelotMode_group.checkedButton()
        if checked_button:
            self.params['camelot_mode'] = checked_button.text().lower()
        # Update pages parameter
        if self.pages_all_radio.isChecked():
            self.params['pages'] = 'all'
        else:
            self.params['pages'] = self.pages_input.text().strip()
        self.params.update({
            'export_format': self.cmb_export.currentText().lower()
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
            'camelot_mode': self.params['camelot_mode'],
            'export_format': self.cmb_export.currentText().lower(),
            'output_path': self.lbl_output.text(),
            'pages': pages_value,
            'current_filepath': self.params.get('current_filepath', '')
        }
