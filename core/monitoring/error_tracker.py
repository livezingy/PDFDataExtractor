# core/monitoring/error_tracker.py
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
from core.utils.logger import AppLogger

@dataclass
class ErrorRecord:
    """Error record data class"""
    timestamp: str
    file_path: str
    error_type: str
    message: str
    stack_trace: str

class ErrorTracker:
    """Error tracking class
    
    Responsible for tracking and recording errors during processing, including:
    1. Error recording
    2. Error reporting
    3. Stack trace collection
    """
    
    def __init__(self):
        """Initialize error tracker"""
        self.errors: List[ErrorRecord] = []
        self.logger = AppLogger()
        self.logger.log_operation("Initialize error tracker", {})

    def record_error(self, file_path: str, exception: Exception):
        """Record error information
        
        Args:
            file_path: File path where error occurred
            exception: Exception object
        """
        record = ErrorRecord(
            timestamp=datetime.now().isoformat(),
            file_path=file_path,
            error_type=type(exception).__name__,
            message=str(exception),
            stack_trace=self._get_stack_trace(exception)
        )
        self.errors.append(record)
        self.logger.log_exception(exception, {
            "file_path": file_path,
            "error_type": type(exception).__name__,
            "timestamp": record.timestamp
        })

    def get_error_report(self) -> List[Dict]:
        """Generate error report
        
        Returns:
            List of error records
        """
        self.logger.log_operation("Generate error report", {
            "error_count": len(self.errors)
        })
        return [{
            'timestamp': err.timestamp,
            'file': err.file_path,
            'error_type': err.error_type,
            'message': err.message
        } for err in self.errors]

    @staticmethod
    def _get_stack_trace(exception: Exception) -> str:
        """Get stack trace from exception
        
        Args:
            exception: Exception object
            
        Returns:
            Formatted stack trace string
        """
        import traceback
        return ''.join(traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ))