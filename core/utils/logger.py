# core/utils/logger.py
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
import os
import sys
import traceback
from typing import Optional, Dict, Any
from PySide6.QtCore import QObject, Signal
from core.utils.path_utils import get_output_paths
from config.paths import get_output_structure

class LogSignals(QObject):
    """Signals for log messages"""
    log_message = Signal(str)  # Signal for log messages

class AppLogger:
    """Application log manager
    
    Provides centralized logging functionality.
    Supports file and console logging with rotation.
    """
    
    _instance = None
    _logger = None
    _signals = LogSignals()
    
    @classmethod
    def get_logger(cls) -> 'AppLogger':
        """Get logger instance
        
        Returns:
            AppLogger instance
        """
        if cls._instance is None:
            cls._instance = cls()
        if cls._logger is None:
            cls._instance._init_logger()
        return cls._instance
        
    def __init__(self):
        """Initialize logger"""
        if AppLogger._instance is not None:
            raise Exception("This class is a singleton!")
        AppLogger._instance = self
        # set default output_path
        if getattr(sys, 'frozen', False):
            # package mode, sys.executable is the main executable
            default_output = os.path.join(os.path.dirname(sys.executable), 'output')
        else:
            # the same directory as the script
            default_output = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'output')
        self.config = {'output_path': default_output}
        self._init_logger()
            
    def _init_logger(self):
        """Initialize logger configuration"""
        # Create logger
        self._logger = logging.getLogger('PDFExtractor')
        self._logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # Create file handler
        # Get output paths from configuration
        if hasattr(self, 'config') and 'output_path' in self.config and self.config['output_path']:
            # Use user specified output path
            try:
                paths = get_output_paths(self.config['output_path'])
                log_dir = paths['debug']
            except Exception:
                # fallback if get_output_paths fails
                log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        else:
            # Fallback to default logs directory
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        # Ensure log directory exists
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'app.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)
        
        # Set class logger
        AppLogger._logger = self._logger
        
    def debug(self, message: str, data: Optional[Dict] = None):
        """Log debug message
        
        Args:
            message: Log message
            data: Additional data
        """
        if data:
            message = f"{message} - {data}"
        self._logger.debug(message)
        
    def info(self, message: str, data: Optional[Dict] = None):
        """Log info message
        
        Args:
            message: Log message
            data: Additional data
        """
        if data:
            message = f"{message} - {data}"
        self._logger.info(message)
        self._signals.log_message.emit(f"INFO: {message}")
        
    def warning(self, message: str, data: Optional[Dict] = None):
        """Log warning message
        
        Args:
            message: Log message
            data: Additional data
        """
        if data:
            message = f"{message} - {data}"
        self._logger.warning(message)
        self._signals.log_message.emit(f"WARNING: {message}")
        
    def error(self, message: str, exc_info: bool = False):
        """Log error message
        
        Args:
            message: Error message
            exc_info: Whether to include exception info
        """
        if exc_info:
            self._logger.error(message, exc_info=True)
            self._signals.log_message.emit(f"ERROR: {message}")
        else:
            self._logger.error(message)
            self._signals.log_message.emit(f"ERROR: {message}")
        
    def critical(self, message: str, data: Optional[Dict] = None):
        """Log critical message
        
        Args:
            message: Log message
            data: Additional data
        """
        if data:
            message = f"{message} - {data}"
        self._logger.critical(message)
        self._signals.log_message.emit(f"CRITICAL: {message}")
        
    def log_exception(self, exception: Exception, data: Optional[Dict] = None):
        """Log exception
        
        Args:
            exception: Exception object
            data: Additional data
        """
        message = f"Exception: {str(exception)}"
        if data:
            message = f"{message} - {data}"
        self._logger.exception(message)
        self._signals.log_message.emit(f"ERROR: {message}")
        
    def log_operation(self, operation: str, data: Optional[Dict] = None):
        """Log operation
        
        Args:
            operation: Operation name
            data: Additional data
        """
        message = f"Operation: {operation}"
        if data:
            message = f"{message} - {data}"
        self._logger.info(message)
        self._signals.log_message.emit(f"INFO: {message}")
        
    def log_performance(self, operation: str, duration: float, data: Optional[Dict] = None):
        """Log performance
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            data: Additional data
        """
        message = f"Performance: {operation} - {duration:.2f}s"
        if data:
            message = f"{message} - {data}"
        self._logger.info(message)
        self._signals.log_message.emit(f"INFO: {message}")
        
    @classmethod
    def get_signals(cls) -> LogSignals:
        """Get log signals
        
        Returns:
            LogSignals instance
        """
        return cls._signals
        
    def set_output_path(self, output_path: str):
        """Set output path and reinitialize logger
        
        Args:
            output_path: Path to output directory
        """
        # set output path in config when output_path is changed
        if self.config.get('output_path') != output_path:
            self.config['output_path'] = output_path
            self._init_logger()  # Reinitialize logger with new path