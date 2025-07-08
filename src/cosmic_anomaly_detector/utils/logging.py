"""
Logging Configuration for Cosmic Anomaly Detector

Provides centralized logging setup with scientific computing best practices.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ScientificFormatter(logging.Formatter):
    """Custom formatter for scientific logging with structured output"""
    
    def __init__(self, include_metadata: bool = True):
        super().__init__()
        self.include_metadata = include_metadata
        
    def format(self, record: logging.LogRecord) -> str:
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra scientific metadata if available
        if self.include_metadata and hasattr(record, 'scientific_context'):
            log_entry['scientific_context'] = record.scientific_context
            
        if hasattr(record, 'processing_stage'):
            log_entry['processing_stage'] = record.processing_stage
            
        if hasattr(record, 'image_id'):
            log_entry['image_id'] = record.image_id
            
        if hasattr(record, 'detection_confidence'):
            log_entry['detection_confidence'] = record.detection_confidence
            
        # Handle exceptions
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, indent=None)


class LoggingManager:
    """Manages logging configuration for the cosmic anomaly detector"""
    
    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self._configured = False
        
    def setup_logging(
        self, 
        log_level: str = "INFO",
        log_format: Optional[str] = None,
        log_dir: str = "logs",
        console_output: bool = True,
        file_output: bool = True,
        structured_output: bool = False
    ) -> None:
        """Setup logging configuration for the system"""
        
        if self._configured:
            return
            
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Setup formatters
        if structured_output:
            formatter = ScientificFormatter()
            console_formatter = ScientificFormatter(include_metadata=False)
        else:
            default_format = (
                log_format or 
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(funcName)s:%(lineno)d - %(message)s"
            )
            formatter = logging.Formatter(default_format)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S"
            )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
            
        # File handlers
        if file_output:
            # Main log file with rotation
            main_log_file = log_path / "cosmic_anomaly_detector.log"
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            # Error log file
            error_log_file = log_path / "errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)
            
            # Scientific analysis log (for reproducibility)
            analysis_log_file = log_path / "scientific_analysis.log"
            analysis_handler = logging.handlers.RotatingFileHandler(
                analysis_log_file,
                maxBytes=20*1024*1024,  # 20MB
                backupCount=10
            )
            analysis_handler.setLevel(logging.INFO)
            analysis_handler.setFormatter(formatter)
            
            # Create scientific logger
            scientific_logger = logging.getLogger("scientific")
            scientific_logger.addHandler(analysis_handler)
            scientific_logger.setLevel(logging.INFO)
            scientific_logger.propagate = False
            
        self._configured = True
        
        # Log the setup completion
        logger = logging.getLogger(__name__)
        logger.info("Logging system initialized successfully")
        logger.info(f"Log level: {log_level}")
        logger.info(f"Log directory: {log_path.absolute()}")
        
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific component"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def get_scientific_logger(self) -> logging.Logger:
        """Get the scientific analysis logger for reproducibility"""
        return logging.getLogger("scientific")
        
    def log_scientific_event(
        self,
        message: str,
        context: Dict[str, Any],
        level: str = "INFO"
    ) -> None:
        """Log a scientific event with metadata for reproducibility"""
        scientific_logger = self.get_scientific_logger()
        
        # Create log record with scientific context
        record = logging.LogRecord(
            name="scientific",
            level=getattr(logging, level.upper()),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.scientific_context = context
        
        scientific_logger.handle(record)
        
    def log_processing_stage(
        self,
        stage: str,
        image_id: str,
        details: Dict[str, Any],
        level: str = "INFO"
    ) -> None:
        """Log a processing stage for pipeline tracking"""
        logger = self.get_logger("processing")
        
        record = logging.LogRecord(
            name="processing",
            level=getattr(logging, level.upper()),
            pathname="",
            lineno=0,
            msg=f"Processing stage: {stage}",
            args=(),
            exc_info=None
        )
        record.processing_stage = stage
        record.image_id = image_id
        record.scientific_context = details
        
        logger.handle(record)
        
    def log_detection_event(
        self,
        detection_type: str,
        confidence: float,
        image_id: str,
        coordinates: Dict[str, float],
        level: str = "INFO"
    ) -> None:
        """Log a detection event with all relevant scientific metadata"""
        logger = self.get_logger("detection")
        
        context = {
            "detection_type": detection_type,
            "coordinates": coordinates,
            "timestamp": datetime.now().isoformat()
        }
        
        record = logging.LogRecord(
            name="detection",
            level=getattr(logging, level.upper()),
            pathname="",
            lineno=0,
            msg=f"Detection: {detection_type} (confidence: {confidence:.3f})",
            args=(),
            exc_info=None
        )
        record.image_id = image_id
        record.detection_confidence = confidence
        record.scientific_context = context
        
        logger.handle(record)


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager


def setup_logging(**kwargs) -> None:
    """Setup logging with the global manager"""
    manager = get_logging_manager()
    manager.setup_logging(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return get_logging_manager().get_logger(name)


def log_scientific_event(
    message: str,
    context: Dict[str, Any],
    level: str = "INFO"
) -> None:
    """Log a scientific event for reproducibility"""
    get_logging_manager().log_scientific_event(message, context, level)


def log_processing_stage(
    stage: str,
    image_id: str,
    details: Dict[str, Any],
    level: str = "INFO"
) -> None:
    """Log a processing stage"""
    get_logging_manager().log_processing_stage(stage, image_id, details, level)


def log_detection_event(
    detection_type: str,
    confidence: float,
    image_id: str,
    coordinates: Dict[str, float],
    level: str = "INFO"
) -> None:
    """Log a detection event"""
    get_logging_manager().log_detection_event(
        detection_type, confidence, image_id, coordinates, level
    )
