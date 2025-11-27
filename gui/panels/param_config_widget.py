# gui/panels/param_config_widget.py
"""
Parameter Configuration Widget

Provides UI for configuring PDFPlumber and Camelot extraction parameters
with three modes: default, auto, and custom.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                              QComboBox, QLabel, QDoubleSpinBox, QSpinBox, 
                              QGroupBox, QScrollArea, QMessageBox)
from PySide6.QtCore import Signal, Qt
from core.utils.param_config import (
    PDFPLUMBER_PARAM_DEFS, CAMELOT_LATTICE_PARAM_DEFS, CAMELOT_STREAM_PARAM_DEFS,
    validate_pdfplumber_params, validate_camelot_lattice_params, validate_camelot_stream_params,
    get_default_pdfplumber_params, get_default_camelot_lattice_params, get_default_camelot_stream_params
)


class ParamConfigWidget(QWidget):
    """Parameter configuration widget"""
    
    paramsChanged = Signal(dict)
    
    def __init__(self, method: str = 'pdfplumber', parent=None):
        super().__init__(parent)
        self.method = method.lower()
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Parameter mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Parameter Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Default", "Auto", "Custom"])
        self.mode_combo.setCurrentIndex(1)  # Default to Auto
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        # Note: Custom mode will be disabled when flavor is auto (handled by parent)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # Parameter inputs (initially hidden for default/auto modes)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVisible(False)
        
        self.params_widget = QWidget()
        self.params_layout = QFormLayout()
        self.params_widget.setLayout(self.params_layout)
        
        self.scroll_area.setWidget(self.params_widget)
        layout.addWidget(self.scroll_area)
        
        # Initialize parameter inputs
        self._init_param_inputs()
        
        self.setLayout(layout)
    
    def _init_param_inputs(self):
        """Initialize parameter input widgets"""
        # Clear existing widgets
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.param_widgets = {}
        
        # Get parameter definitions based on method and flavor
        if not hasattr(self, 'param_defs'):
            if self.method == 'pdfplumber':
                self.param_defs = PDFPLUMBER_PARAM_DEFS
            elif self.method == 'camelot':
                # Will be determined by flavor in set_method
                self.param_defs = {}
            else:
                self.param_defs = {}
        
        param_defs = self.param_defs
        
        for param_name, param_def in param_defs.items():
            label = QLabel(param_name.replace('_', ' ').title() + ":")
            label.setToolTip(param_def.get('description', ''))
            
            if param_def['type'] == int:
                widget = QSpinBox()
                if 'range' in param_def:
                    widget.setRange(int(param_def['range'][0]), int(param_def['range'][1]))
                widget.setValue(int(param_def['default']))
            elif param_def['type'] == float:
                widget = QDoubleSpinBox()
                if 'range' in param_def:
                    widget.setRange(param_def['range'][0], param_def['range'][1])
                widget.setValue(param_def['default'])
                widget.setSingleStep(0.1)
            elif param_def['type'] == str:
                if 'options' in param_def:
                    widget = QComboBox()
                    widget.addItems(param_def['options'])
                    widget.setCurrentText(param_def['default'])
                else:
                    from PySide6.QtWidgets import QLineEdit
                    widget = QLineEdit()
                    widget.setText(str(param_def['default']))
            else:
                continue
            
            widget.setToolTip(param_def.get('description', ''))
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self._on_param_changed)
            elif hasattr(widget, 'currentIndexChanged'):
                widget.currentIndexChanged.connect(self._on_param_changed)
            elif hasattr(widget, 'textChanged'):
                widget.textChanged.connect(self._on_param_changed)
            
            self.param_widgets[param_name] = widget
            self.params_layout.addRow(label, widget)
    
    def set_method(self, method: str, flavor: str = None):
        """Set extraction method and flavor"""
        self.method = method.lower()
        self.flavor = flavor.lower() if flavor else None
        
        if method.lower() == 'camelot':
            if flavor and flavor.lower() == 'lattice':
                self.param_defs = CAMELOT_LATTICE_PARAM_DEFS
            elif flavor and flavor.lower() == 'stream':
                self.param_defs = CAMELOT_STREAM_PARAM_DEFS
            else:
                self.param_defs = {}
        else:
            self.param_defs = PDFPLUMBER_PARAM_DEFS if method.lower() == 'pdfplumber' else {}
        
        self._init_param_inputs()
        self._on_mode_changed()
    
    def _on_mode_changed(self):
        """Handle mode change"""
        mode = self.mode_combo.currentText().lower()
        self.scroll_area.setVisible(mode == 'custom')
        self._update_params()
    
    def _on_param_changed(self):
        """Handle parameter change"""
        if self.mode_combo.currentText().lower() == 'custom':
            self._update_params()
    
    def _update_params(self):
        """Update parameters and emit signal"""
        mode = self.mode_combo.currentText().lower()
        
        if mode == 'default':
            if self.method == 'pdfplumber':
                params = get_default_pdfplumber_params()
            elif self.method == 'camelot':
                # Get flavor from parent or use default
                if hasattr(self, 'flavor'):
                    if self.flavor == 'lattice':
                        params = get_default_camelot_lattice_params()
                    elif self.flavor == 'stream':
                        params = get_default_camelot_stream_params()
                    else:
                        params = {}
                else:
                    params = {}
            else:
                params = {}
        elif mode == 'auto':
            params = {}  # Auto mode uses calculated parameters
        else:  # custom
            params = {}
            for param_name, widget in self.param_widgets.items():
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText()
                else:
                    from PySide6.QtWidgets import QLineEdit
                    if isinstance(widget, QLineEdit):
                        try:
                            params[param_name] = float(widget.text())
                        except ValueError:
                            params[param_name] = widget.text()
            
            # For Camelot, ensure flavor is included in custom params
            if self.method == 'camelot' and hasattr(self, 'flavor') and self.flavor:
                params['flavor'] = self.flavor
        
        # Validate parameters
        if mode == 'custom' and params:
            # Store flavor before validation (validation functions may remove it)
            camelot_flavor = None
            if self.method == 'camelot' and 'flavor' in params:
                camelot_flavor = params['flavor']
            
            if self.method == 'pdfplumber':
                is_valid, error_msg, validated_params = validate_pdfplumber_params(params)
            elif self.method == 'camelot':
                if hasattr(self, 'flavor'):
                    if self.flavor == 'lattice':
                        is_valid, error_msg, validated_params = validate_camelot_lattice_params(params)
                    elif self.flavor == 'stream':
                        is_valid, error_msg, validated_params = validate_camelot_stream_params(params)
                    else:
                        is_valid, error_msg, validated_params = True, None, params
                else:
                    is_valid, error_msg, validated_params = True, None, params
            else:
                is_valid, error_msg, validated_params = True, None, params
            
            # Restore flavor after validation
            if self.method == 'camelot' and camelot_flavor:
                validated_params['flavor'] = camelot_flavor
            
            if not is_valid and error_msg:
                QMessageBox.warning(self, "Parameter Validation Error", error_msg)
                params = validated_params
        
        # Emit signal with mode and params
        result = {
            'mode': mode,
            'params': params
        }
        self.paramsChanged.emit(result)
    
    def get_params(self) -> dict:
        """Get current parameters"""
        mode = self.mode_combo.currentText().lower()
        
        if mode == 'default':
            if self.method == 'pdfplumber':
                return {'mode': 'default', 'params': get_default_pdfplumber_params()}
            elif self.method == 'camelot':
                if hasattr(self, 'flavor'):
                    if self.flavor == 'lattice':
                        return {'mode': 'default', 'params': get_default_camelot_lattice_params()}
                    elif self.flavor == 'stream':
                        return {'mode': 'default', 'params': get_default_camelot_stream_params()}
            return {'mode': 'default', 'params': {}}
        elif mode == 'auto':
            return {'mode': 'auto', 'params': {}}
        else:  # custom
            params = {}
            for param_name, widget in self.param_widgets.items():
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText()
                else:
                    from PySide6.QtWidgets import QLineEdit
                    if isinstance(widget, QLineEdit):
                        try:
                            params[param_name] = float(widget.text())
                        except ValueError:
                            params[param_name] = widget.text()
            
            # For Camelot, ensure flavor is included in custom params
            if self.method == 'camelot' and hasattr(self, 'flavor') and self.flavor:
                params['flavor'] = self.flavor
            
            # For Camelot, ensure flavor is included in custom params
            if self.method == 'camelot' and hasattr(self, 'flavor') and self.flavor:
                validated_params['flavor'] = self.flavor
            
            # Validate
            if self.method == 'pdfplumber':
                is_valid, error_msg, validated_params = validate_pdfplumber_params(params)
            elif self.method == 'camelot':
                if hasattr(self, 'flavor'):
                    if self.flavor == 'lattice':
                        is_valid, error_msg, validated_params = validate_camelot_lattice_params(params)
                    elif self.flavor == 'stream':
                        is_valid, error_msg, validated_params = validate_camelot_stream_params(params)
                    else:
                        is_valid, error_msg, validated_params = True, None, params
                else:
                    is_valid, error_msg, validated_params = True, None, params
            else:
                is_valid, error_msg, validated_params = True, None, params
            
            # Ensure flavor is preserved after validation
            if self.method == 'camelot' and hasattr(self, 'flavor') and self.flavor:
                validated_params['flavor'] = self.flavor
            
            return {'mode': 'custom', 'params': validated_params}
    
    def set_params(self, mode: str, params: dict):
        """Set parameters"""
        # Set mode
        mode_index = self.mode_combo.findText(mode.capitalize(), Qt.MatchFixedString)
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)
        
        # Set parameter values
        if mode == 'custom' and params:
            for param_name, value in params.items():
                if param_name in self.param_widgets:
                    widget = self.param_widgets[param_name]
                    if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                        widget.setValue(value)
                    elif isinstance(widget, QComboBox):
                        widget.setCurrentText(str(value))
                    else:
                        from PySide6.QtWidgets import QLineEdit
                        if isinstance(widget, QLineEdit):
                            widget.setText(str(value))

