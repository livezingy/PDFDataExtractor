# gui/panels/progress_panel.py
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,QProgressBar,QTabWidget, QGraphicsView, QGraphicsScene,
                              QTextEdit, QPushButton, QLabel, QCheckBox, QComboBox,QListWidget,QListWidgetItem)
from PySide6.QtGui import QKeySequence, QPixmap, QPainter, QPen, QColor, QShortcut
from PySide6.QtCore import Signal, Qt,QEvent
import glob
import re, os
import time
from core.utils.logger import AppLogger
from core.utils.path_utils import get_output_subpath

class ImageViewer(QGraphicsView):
    """Image viewer for previewing detected tables
    
    Supports page navigation and image caching.
    """
    
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.current_page = 0
        self.total_pages = 0
        self.image_paths = []
        self.current_file = ""
        self.image_cache = {} 
        self.setToolTip("Use Left/Right arrow keys to navigate pages")
 
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.prev_page)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.next_page)

  

    def show_current_page(self):
        """Display current page"""
        self.scene.clear()
        if self.current_page < len(self.image_paths):
            path = self.image_paths[self.current_page]
            if path not in self.image_cache:
                self.image_cache[path] = QPixmap(path)
            pixmap = self.image_cache[path]
            self.scene.addPixmap(pixmap)
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def next_page(self):
        """Show next page"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.show_current_page()

    def prev_page(self):
        """Show previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.show_current_page()

    def keyPressEvent(self, event):
        """Handle key press events
        
        Args:
            event: Key press event
        """
        if event.key() == Qt.Key_Left:
            self.prev_page()
        elif event.key() == Qt.Key_Right:
            self.next_page()
        else:
            super().keyPressEvent(event)


    def clear_cache(self):
        """Clear image cache and release QPixmap resources"""
        self.scene.clear()
        for pixmap in self.image_cache.values():
            del pixmap
        self.image_cache.clear()
        self.image_paths = []
        self.current_page = 0
        self.total_pages = 0
        self.current_file = ""

class ProgressPanel(QWidget):
    """Progress display panel
    
    Panel for displaying processing progress and status.
    Shows progress bars and file status.
    """
    
    def __init__(self):
        """Initialize progress panel"""
        super().__init__()
        self.logger = AppLogger.get_logger()
        self.initial_status = "Ready to process"
        self.preview_enabled = False
        self.current_pdf_path = None  # 记录当前PDF
        self._init_ui()
        self._init_tabs()
        self._connect_signals()
        self.params = {}
        self.tab_widget.currentChanged.connect(self._on_tab_changed)  # 监听Tab切换
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        self.lbl_status = QLabel("Ready to process")

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.log_view.setStyleSheet("""
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                font-family: "Segoe UI";
                font-size: 9pt;
                color: #333333;
            }
        """)
        
        # Set panel style
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f4f8;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 3px;
                text-align: center;
                background-color: #f8f9fa;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: white;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
            }
            QLabel {
                font-weight: bold;
            }
        """)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Add status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Add file list
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)
        
        # Control row: Checkbox + button
        self.control_row = QWidget()
        self.chk_preview = QCheckBox("Preview Detected Tables")
        
        # Horizontal layout
        h_layout = QHBoxLayout(self.control_row)
        h_layout.addWidget(self.chk_preview)
        h_layout.addStretch()  # Add elastic space
        h_layout.setContentsMargins(0, 0, 0, 0)
        
    def _init_tabs(self):
        """Initialize tab structure"""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # --- Control row always visible ---
        main_layout.addWidget(self.control_row)
        
        # --- Tab container ---
        self.tab_widget = QTabWidget()
        
        # Progress tab
        self.progress_tab = QWidget()
        progress_layout = QVBoxLayout(self.progress_tab)
        progress_layout.addWidget(self.lbl_status)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(QLabel("Processing Log:"))
        progress_layout.addWidget(self.log_view)
        
        # Images tab (always exists but initially empty)
        self.image_tab = QWidget()
        self.image_viewer = ImageViewer()  # Assume already defined

        # --- Navigation buttons row ---
        control_widget = QWidget()
        btn_prev = QPushButton("← Previous")
        btn_next = QPushButton("Next →")
        btn_prev.setFixedHeight(40)
        btn_next.setFixedHeight(40)
        btn_prev.setMinimumWidth(100)
        btn_next.setMinimumWidth(100)
        nav_layout = QHBoxLayout(control_widget)
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        nav_layout.addStretch()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        # 绑定事件
        btn_prev.clicked.connect(self.image_viewer.prev_page)
        btn_next.clicked.connect(self.image_viewer.next_page)
        # --- Image/No-image area ---
        image_layout = QVBoxLayout(self.image_tab)
        image_layout.addWidget(control_widget)
        # 使用QScrollArea包裹image_viewer，支持大图滚动
        from PySide6.QtWidgets import QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_viewer.setMinimumHeight(360)
        self.image_viewer.setMinimumWidth(400)
        self.scroll_area.setWidget(self.image_viewer)
        self.no_images_label = QLabel("No images available")
        self.no_images_label.setAlignment(Qt.AlignCenter)
        self.no_images_label.setStyleSheet("color: #888; font-size: 14px; font-weight: normal;")
        image_layout.addWidget(self.no_images_label)
        image_layout.addWidget(self.scroll_area)
        self.no_images_label.setVisible(True)
        self.scroll_area.setVisible(False)
        image_layout.setStretch(0, 0)  # control_widget
        image_layout.setStretch(1, 1)  # no_images_label
        image_layout.setStretch(2, 1)  # scroll_area
        
        # Add tabs
        self.tab_widget.addTab(self.progress_tab, "Progress")
        self.tab_widget.addTab(self.image_tab, "Images")
        
        # Initially hide Images tab
        self.tab_widget.setTabVisible(1, False)
        
        main_layout.addWidget(self.tab_widget)

    def _connect_signals(self):
        """Connect signals"""
        self.chk_preview.stateChanged.connect(self._toggle_preview)
        # Connect logger signals
        AppLogger.get_signals().log_message.connect(self._handle_log_message)
        # Connect context menu
        self.log_view.customContextMenuRequested.connect(self._show_log_context_menu)

    def _handle_log_message(self, message: str):
        """Handle log message
        
        Args:
            message: Log message
        """
        self.log_view.append(message)
        # Auto scroll to bottom
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def _show_log_context_menu(self, position):
        """Show context menu for log view
        
        Args:
            position: Mouse position
        """
        from PySide6.QtWidgets import QMenu
        
        menu = QMenu(self)
        clear_action = menu.addAction("清空日志")
        clear_action.triggered.connect(self.log_view.clear)
        menu.exec(self.log_view.mapToGlobal(position))

    def _toggle_preview(self, state):
        """Toggle Images tab visibility
        
        Args:
            state: Checkbox state
        """
        if state == 2:  # Checked
            self.tab_widget.setTabVisible(1, True)
            self.preview_enabled = True
        else:
            self.tab_widget.setTabVisible(1, False)
            self.preview_enabled = False
            self.image_viewer.scene.clear()  # Clear image viewer

    def update_progress(self, value, message):
        """Update progress display
        
        Args:
            value: Progress value
            message: Progress message
        """
        self.progress_bar.setValue(int(value))
        self.log_view.append(f"Progress: {value}% - {message}")  

    def update_status(self, status_text: str):
        """Update status display
        
        Args:
            status_text: Status text
        """
        self.lbl_status.setText(status_text)

    def update_progress(self, percentage: int, message: str):
        """Update progress
        
        Args:
            percentage: Progress percentage
            message: Status message
        """
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

    def add_log_message(self, message: str):
        """Add log message with timestamp
        
        Args:
            message: Log message
        """
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {message}")
        # Auto scroll to bottom
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def reset_progress(self):
        """Reset progress display"""
        self.progress_bar.setValue(0)
        self.lbl_status.clear()
        self.log_view.clear()
        
    def update_file_status(self, file_path: str, success: bool):
        """Update file status
        
        Args:
            file_path: File path
            success: Whether processing was successful
        """
        item = QListWidgetItem()
        if success:
            item.setText(f"✓ {file_path}")
            item.setForeground(Qt.green)
            self.current_pdf_path = file_path  # record current PDF path
            if self.preview_enabled:
                self._load_preview_images_for_pdf()
        else:
            item.setText(f"✗ {file_path}")
            item.setForeground(Qt.red)
        self.file_list.addItem(item)
        
    def show_error(self, file_path: str, error_message: str):
        """Show error message
        
        Args:
            file_path: File path
            error_message: Error message
        """
        item = QListWidgetItem()
        item.setText(f"✗ {file_path} - {error_message}")
        item.setForeground(Qt.red)
        self.file_list.addItem(item)
        
    def clear(self):
        """Clear progress and status"""
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.file_list.clear()
        self.image_viewer.clear_cache()

    def _on_tab_changed(self, idx):
        # Images Tab
        if idx == 1 and self.current_pdf_path and self.preview_enabled:
            self._load_preview_images_for_pdf()
    

    def _load_preview_images_for_pdf(self):
        import os, glob, re
        preview_dir = get_output_subpath(self.params, 'preview')
        image_paths = sorted(
            glob.glob(os.path.join(preview_dir, f"page*_detection.png")),
            key=lambda x: int(re.search(r'page(\\d+)_detection', os.path.basename(x)).group(1)) if re.search(r'page(\\d+)_detection', os.path.basename(x)) else 0
        )
        self.image_viewer.image_paths = image_paths
        self.image_viewer.total_pages = len(image_paths)
        if self.image_viewer.total_pages > 0:
            self.no_images_label.setVisible(False)
            self.scroll_area.setVisible(True)
            self.image_viewer.setVisible(True)
            self.image_viewer.current_page = 0
            self.image_viewer.show_current_page()
        else:
            self.no_images_label.setVisible(True)
            self.scroll_area.setVisible(False)
            self.image_viewer.setVisible(False)
            self.image_viewer.clear_cache()
