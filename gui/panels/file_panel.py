# gui/panels/file_panel.py
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QFileDialog, QListWidget, QListWidgetItem)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from core.utils.logger import AppLogger
import os


class DropLabel(QLabel):
    """Custom label for file drop area
    
    Supports both drag & drop and click events.
    """
    
    def __init__(self, parent=None):
        """Initialize drop label"""
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)  # Change cursor on hover
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press event
        
        Args:
            event: Mouse press event
        """
        if event.button() == Qt.LeftButton:
            self.parent()._add_files()


class FilePanel(QWidget):
    """File selection panel
    
    Panel for selecting and managing input files.
    Provides drag and drop support.
    """
    
    files_added = Signal(list)  # Signal emitted when files are added
    
    def __init__(self):
        """Initialize file panel"""
        super().__init__()
        self.logger = AppLogger.get_logger()
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        self.setWindowTitle("File Selection")
        
        # Create drop area
        self.lbl_drop = DropLabel("Drag & Drop PDF Files Here\nor Click to Select")
        self.lbl_drop.setAlignment(Qt.AlignCenter)
        self.lbl_drop.setStyleSheet("""
            QLabel {
                border: 2px dashed #4A90E2;
                border-radius: 10px;
                padding: 30px;
                color: #666;
                font-size: 14px;
                background-color: #f8f9fa;
            }
            QLabel:hover {
                background-color: #e3f2fd;
                border-color: #2196F3;
            }
        """)
        self.lbl_drop.setAcceptDrops(True)
        self.lbl_drop.dragEnterEvent = self._drag_enter_event
        self.lbl_drop.dropEvent = self._drop_event
        layout.addWidget(self.lbl_drop)
        
        # Add file list
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
            }
        """)
        layout.addWidget(self.file_list)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("Add Files")        
        self.btn_add.clicked.connect(self._add_files)
        button_layout.addWidget(self.btn_add)
        
        self.btn_clear = QPushButton("Clear")       
        self.btn_clear.clicked.connect(self._clear_files)
        button_layout.addWidget(self.btn_clear)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def _add_files(self):
        """Add files through file dialog"""
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Files",
                "",
                "PDF/Image Files (*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)"
            )
            
            if files:
                self._add_files_to_list(files)
                
            self.logger.log_operation("File addition", {
                "file_count": len(files)
            })
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "File addition"
            })
            
    def _add_files_to_list(self, files: list):
        """Add files to list widget
        
        Args:
            files: List of file paths
        """
        for file in files:
            if file not in self._get_file_list():
                item = QListWidgetItem(os.path.basename(file))
                item.setToolTip(file)
                self.file_list.addItem(item)
                
        self.files_added.emit(self._get_file_list())
        
    def _clear_files(self):
        """Clear file list"""
        self.file_list.clear()
        self.files_added.emit([])

        
    def _get_file_list(self) -> list:
        """Get list of file paths
        
        Returns:
            List of file paths
        """
        return [self.file_list.item(i).toolTip() 
                for i in range(self.file_list.count())]
        
    def _drag_enter_event(self, event):
        """Handle drag enter event
        
        Args:
            event: Drag enter event
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def _drop_event(self, event):
        """Handle drop event
        
        Args:
            event: Drop event
        """
        files = []
        for url in event.mimeData().urls():
            file = url.toLocalFile()
            if file.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                files.append(file)
                
        if files:
            self._add_files_to_list(files)