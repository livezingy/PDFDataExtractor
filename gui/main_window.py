# gui/main_window.py
import matplotlib
matplotlib.use('Qt5Agg')  # Set matplotlib backend

from PySide6.QtCore import QThreadPool, QRunnable, Slot, Signal, QObject, QThread, Qt, QTimer
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QMessageBox, QVBoxLayout, QPushButton, QLabel)
from gui.panels.file_panel import FilePanel
from gui.panels.params_panel import ParamsPanel
from gui.panels.progress_panel import ProgressPanel
from core.processing.page_processor import PageProcessor
from core.processing.table_processor import TableProcessor
from core.utils.file_utils import validate_writable
from core.utils.logger import AppLogger
from core.utils.config import Config
from core.models.table_parser import TableParser
import os
import time
import uuid
from typing import Any, Optional, Dict, List, Tuple
import numpy as np
import pdfplumber
import gc


class ProcessingSignals(QObject):
    progress_updated = Signal(int, str)  # (percentage, message)
    file_complete = Signal(str, bool)    # (file path, success)
    error_occurred = Signal(str, Exception)
    status_updated = Signal(dict)        # Status update

class FileTask(QRunnable):
    def __init__(self, task_id: str, params: dict):
        super().__init__()
        self.task_id = task_id
        self.params = params
        self.signals = ProcessingSignals()
        self.logger = AppLogger.get_logger()
        
    @Slot()
    def run(self):
        try:
            # Initialize page processor
            page_processor = PageProcessor(self.params)

            # Process file using page processor
            import asyncio
            results = asyncio.run(page_processor.process(self.params))

            if results['success']:
                # Update progres
                self.signals.progress_updated.emit(100, "Processing completed")
                self.signals.file_complete.emit(self.params.get("current_filepath"), True)
                
                self.logger.info(f"File processing completed: {self.params.get('current_filepath', '')}")
            else:
                self.signals.error_occurred.emit(self.params.get('current_filepath', ''), Exception(results.get('error', 'Unknown error')))
                self.signals.file_complete.emit(self.params.get('current_filepath', ''), False)
            
        except Exception as e:
            self.logger.error(f"File processing failed: {str(e)}", exc_info=True)
            self.signals.error_occurred.emit(self.params.get('current_filepath', ''), e)
            self.signals.file_complete.emit(self.params.get('current_filepath', ''), False)



class MainWindow(QMainWindow):
    """Main window class
    
    Main application window that integrates all panels.
    Manages file processing workflow.
    """

    def __init__(self, app_config):
        """Initialize main window"""
        super().__init__()  
        self.config = app_config
        # Initialize logger with output path from config
        output_path = self.config.get('ui', {}).get('output_path', '')        
        logger_config = {'output_path': output_path} if output_path else {}
        self.logger = AppLogger.get_logger(logger_config)
            
        self.table_parser = TableParser(app_config)
        self.thread_pool = QThreadPool()
        self._init_ui()

    def _init_ui(self):
        """Initialize UI components"""
        # Set window properties
        self.setWindowTitle("PDF Table Extractor")
        self.setMinimumSize(1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        #Right  panel (Progress)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        
        # Left panel (File and Parameters)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)
        
        # Create panels
        self.progress_panel = ProgressPanel()
        self.file_panel = FilePanel()
        self.params_panel = ParamsPanel(self.config)

        # Add panels to layouts
        right_layout.addWidget(self.progress_panel)
        left_layout.addWidget(self.file_panel)
        left_layout.addWidget(self.params_panel)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Connect signals
        self.file_panel.files_added.connect(self._handle_files_added)
        self.params_panel.process_started.connect(self._handle_process_started)
        # 不在这里log_operation，等logger初始化后再log

    def _handle_files_added(self, files: List[str]):
        """Handle file addition
        
        Args:
            files: List of file paths
        """
        try:
            # Update process panel
            #self.params_panel.reset()
            
            """ if self.logger:
                self.logger.log_operation("File addition handling", {
                    "file_count": len(files)
                }) """
            
        except Exception as e:
            self.logger.log_exception(e, {
                "operation": "File addition handling"
            })
            
    def _handle_process_started(self, params: Dict):
        """Handle process start
        
        Args:
            params: Process parameters
        """
        try:
            output_path = params.get('output_path')
            if output_path:
                self.logger.set_output_path(output_path)  
            # Get files from file panel
            files = self.file_panel._get_file_list()
            if not files:
                QMessageBox.warning(self, "Warning", "No files selected")
                return
            params['table_parser'] = self.table_parser  # 注入table_parser实例
            params['models'] = self.table_parser.models
            # Start processing each file
            for file_path in files:
                file_params = params.copy()  # copy parameters for each file
                self.progress_panel.params = file_params
                file_params['current_filepath'] = file_path  # set current file path
                task_id = str(uuid.uuid4())
                task = FileTask(task_id, file_params)
                task.logger = self.logger  # 所有子线程用同一个logger实例
                # Connect signals
                task.signals.progress_updated.connect(
                    lambda p, m: self.progress_panel.update_progress(p, m)
                )
                task.signals.file_complete.connect(
                    lambda f, s: self.progress_panel.update_file_status(f, s)
                )
                task.signals.error_occurred.connect(
                    lambda f, e: self.progress_panel.show_error(f, str(e))
                )
                # Start task
                self.thread_pool.start(task)
            if self.logger:
                self.logger.log_operation("Process start handling", {
                    "params": params,
                    "file_count": len(files)
                })
        except Exception as e:
            if self.logger:
                self.logger.log_exception(e, {
                    "operation": "Process start handling"
                })
    
    def closeEvent(self, event):
        """Handle window close event
        
        Args:
            event: Close event
        """
        try:
            self.thread_pool.waitForDone()
            self.progress_panel.image_viewer.clear_cache()
            if self.logger:
                self.logger.log_operation("Window closing")
            gc.collect()
            event.accept()
        except Exception as e:
            if self.logger:
                self.logger.log_exception(e, {
                    "operation": "Window closing"
                })
            event.accept()



