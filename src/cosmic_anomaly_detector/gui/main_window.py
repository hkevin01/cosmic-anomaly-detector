#!/usr/bin/env python3
"""
Cosmic Anomaly Detector GUI Application

PyQt5-based graphical interface for analyzing JWST images and detecting
artificial structures and anomalies in space.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph import ImageView, PlotWidget

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cosmic_anomaly_detector.core.detector import AnomalyDetector
from cosmic_anomaly_detector.processing.advanced_cv import (
    AdvancedNoiseReducer,
    GeometricAnalyzer,
    ObjectDetector,
    ScaleInvariantFeatureDetector,
)
from cosmic_anomaly_detector.utils.config import SystemConfig, get_config
from cosmic_anomaly_detector.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class ImageAnalysisWorker(QThread):
    """Worker thread for image analysis to prevent GUI freezing"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, image_path: str, config: SystemConfig):
        super().__init__()
        self.image_path = image_path
        self.config = config
        self.detector = AnomalyDetector(config)
        
    def run(self):
        """Run image analysis in background thread"""
        try:
            self.status_updated.emit("Loading image...")
            self.progress_updated.emit(10)
            
            # Initialize components
            noise_reducer = AdvancedNoiseReducer(self.config.image_processing)
            object_detector = ObjectDetector(self.config.image_processing)
            geometric_analyzer = GeometricAnalyzer(self.config.image_processing)
            feature_detector = ScaleInvariantFeatureDetector(self.config.image_processing)
            
            self.status_updated.emit("Processing image...")
            self.progress_updated.emit(30)
            
            # Load and process image
            from astropy.io import fits
            with fits.open(self.image_path) as hdul:
                image_data = hdul[0].data
                header = hdul[0].header
                
                # Get WCS if available
                wcs = None
                try:
                    from astropy.wcs import WCS
                    wcs = WCS(header)
                except:
                    pass
            
            self.status_updated.emit("Reducing noise...")
            self.progress_updated.emit(40)
            
            # Apply noise reduction
            denoised = noise_reducer.adaptive_bilateral_filter(image_data)
            
            self.status_updated.emit("Detecting objects...")
            self.progress_updated.emit(60)
            
            # Detect objects
            objects = object_detector.detect_objects(denoised, wcs)
            
            self.status_updated.emit("Analyzing geometry...")
            self.progress_updated.emit(80)
            
            # Analyze geometry
            geometric_features = geometric_analyzer.analyze_geometry(objects, denoised)
            
            # Detect scale-invariant features
            keypoints, descriptors = feature_detector.detect_features(denoised)
            feature_patterns = feature_detector.analyze_feature_patterns(keypoints, descriptors)
            
            self.status_updated.emit("Analysis complete!")
            self.progress_updated.emit(100)
            
            # Compile results
            results = {
                'original_image': image_data,
                'processed_image': denoised,
                'objects': objects,
                'geometric_features': geometric_features,
                'keypoints': keypoints,
                'feature_patterns': feature_patterns,
                'header': header,
                'wcs': wcs
            }
            
            self.analysis_complete.emit(results)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)


class ImageDisplayWidget(QWidget):
    """Widget for displaying astronomical images with overlays"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_image = None
        self.objects = []
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Image view
        self.image_view = ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(1, 100)
        self.contrast_slider.setValue(50)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        controls_layout.addWidget(QLabel("Contrast:"))
        controls_layout.addWidget(self.contrast_slider)
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-50, 50)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        controls_layout.addWidget(QLabel("Brightness:"))
        controls_layout.addWidget(self.brightness_slider)
        
        self.show_objects_cb = QCheckBox("Show Detected Objects")
        self.show_objects_cb.stateChanged.connect(self.update_overlays)
        controls_layout.addWidget(self.show_objects_cb)
        
        layout.addLayout(controls_layout)
        self.setLayout(layout)
        
    def set_image(self, image: np.ndarray):
        """Set the image to display"""
        self.current_image = image
        self.image_view.setImage(image, autoRange=True, autoLevels=True)
        
    def set_objects(self, objects: List):
        """Set detected objects for overlay"""
        self.objects = objects
        self.update_overlays()
        
    def update_contrast(self):
        """Update image contrast"""
        if self.current_image is not None:
            contrast = self.contrast_slider.value() / 50.0
            brightness = self.brightness_slider.value() / 100.0
            
            adjusted = self.current_image * contrast + brightness
            adjusted = np.clip(adjusted, 0, adjusted.max())
            self.image_view.setImage(adjusted, autoRange=False, autoLevels=False)
            
    def update_brightness(self):
        """Update image brightness"""
        self.update_contrast()  # Same function handles both
        
    def update_overlays(self):
        """Update object detection overlays"""
        if not self.show_objects_cb.isChecked():
            self.image_view.view.clear()
            if self.current_image is not None:
                self.image_view.setImage(self.current_image, autoRange=False)
            return
            
        # Add object overlays
        for obj in self.objects:
            # Draw bounding box
            x1, y1, x2, y2 = obj.bbox
            rect = pg.RectROI([x1, y1], [x2-x1, y2-y1], pen='r')
            rect.setAcceptedMouseButtons(Qt.NoButton)
            self.image_view.view.addItem(rect)
            
            # Add center point
            center = pg.ScatterPlotItem([obj.centroid[1]], [obj.centroid[0]], 
                                      pen='g', brush='g', size=5)
            self.image_view.view.addItem(center)


class ObjectAnalysisWidget(QWidget):
    """Widget for displaying object analysis results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Object table
        self.object_table = QTableWidget()
        self.object_table.setColumnCount(8)
        self.object_table.setHorizontalHeaderLabels([
            "ID", "Centroid", "Area", "Regularity", "Artificial Prob", 
            "Confidence", "RA", "Dec"
        ])
        layout.addWidget(self.object_table)
        
        # Statistics
        stats_group = QGroupBox("Detection Statistics")
        stats_layout = QGridLayout()
        
        self.total_objects_label = QLabel("Total Objects: 0")
        self.high_confidence_label = QLabel("High Confidence: 0")
        self.artificial_candidates_label = QLabel("Artificial Candidates: 0")
        
        stats_layout.addWidget(self.total_objects_label, 0, 0)
        stats_layout.addWidget(self.high_confidence_label, 0, 1)
        stats_layout.addWidget(self.artificial_candidates_label, 1, 0)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        self.setLayout(layout)
        
    def update_objects(self, objects: List, geometric_features: List):
        """Update object table with analysis results"""
        self.object_table.setRowCount(len(objects))
        
        high_confidence_count = 0
        artificial_candidates = 0
        
        for i, (obj, features) in enumerate(zip(objects, geometric_features)):
            # Object ID
            self.object_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            
            # Centroid
            centroid_str = f"({obj.centroid[0]:.1f}, {obj.centroid[1]:.1f})"
            self.object_table.setItem(i, 1, QTableWidgetItem(centroid_str))
            
            # Area
            self.object_table.setItem(i, 2, QTableWidgetItem(f"{obj.area:.1f}"))
            
            # Regularity
            self.object_table.setItem(i, 3, QTableWidgetItem(f"{obj.regularity_score:.3f}"))
            
            # Artificial probability
            artificial_prob = features.artificial_probability
            self.object_table.setItem(i, 4, QTableWidgetItem(f"{artificial_prob:.3f}"))
            
            # Confidence (derived from multiple factors)
            confidence = (obj.regularity_score + features.artificial_probability) / 2
            self.object_table.setItem(i, 5, QTableWidgetItem(f"{confidence:.3f}"))
            
            # Coordinates
            if obj.coordinates:
                ra_str = f"{obj.coordinates.ra.deg:.6f}"
                dec_str = f"{obj.coordinates.dec.deg:.6f}"
            else:
                ra_str = "N/A"
                dec_str = "N/A"
            self.object_table.setItem(i, 6, QTableWidgetItem(ra_str))
            self.object_table.setItem(i, 7, QTableWidgetItem(dec_str))
            
            # Update statistics
            if confidence > 0.7:
                high_confidence_count += 1
            if artificial_prob > 0.8:
                artificial_candidates += 1
                
        # Update statistics labels
        self.total_objects_label.setText(f"Total Objects: {len(objects)}")
        self.high_confidence_label.setText(f"High Confidence: {high_confidence_count}")
        self.artificial_candidates_label.setText(f"Artificial Candidates: {artificial_candidates}")


class ConfigurationWidget(QWidget):
    """Widget for configuring analysis parameters"""
    
    def __init__(self, config: SystemConfig):
        super().__init__()
        self.config = config
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Image processing settings
        img_proc_group = QGroupBox("Image Processing")
        img_proc_layout = QGridLayout()
        
        # Noise reduction
        img_proc_layout.addWidget(QLabel("Noise Reduction Sigma:"), 0, 0)
        self.noise_sigma_spin = QDoubleSpinBox()
        self.noise_sigma_spin.setRange(0.1, 10.0)
        self.noise_sigma_spin.setValue(self.config.image_processing.noise_reduction_sigma)
        self.noise_sigma_spin.setSingleStep(0.1)
        img_proc_layout.addWidget(self.noise_sigma_spin, 0, 1)
        
        # Object detection threshold
        img_proc_layout.addWidget(QLabel("Detection Threshold:"), 1, 0)
        self.detection_threshold_spin = QDoubleSpinBox()
        self.detection_threshold_spin.setRange(0.01, 1.0)
        self.detection_threshold_spin.setValue(self.config.image_processing.edge_detection_threshold)
        self.detection_threshold_spin.setSingleStep(0.01)
        img_proc_layout.addWidget(self.detection_threshold_spin, 1, 1)
        
        # Minimum object area
        img_proc_layout.addWidget(QLabel("Min Object Area:"), 2, 0)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(10, 10000)
        self.min_area_spin.setValue(self.config.image_processing.object_detection_min_area)
        img_proc_layout.addWidget(self.min_area_spin, 2, 1)
        
        img_proc_group.setLayout(img_proc_layout)
        layout.addWidget(img_proc_group)
        
        # Anomaly detection settings
        anomaly_group = QGroupBox("Anomaly Detection")
        anomaly_layout = QGridLayout()
        
        # Detection threshold
        anomaly_layout.addWidget(QLabel("Anomaly Threshold:"), 0, 0)
        self.anomaly_threshold_spin = QDoubleSpinBox()
        self.anomaly_threshold_spin.setRange(0.1, 1.0)
        self.anomaly_threshold_spin.setValue(self.config.anomaly_detection.detection_threshold)
        self.anomaly_threshold_spin.setSingleStep(0.01)
        anomaly_layout.addWidget(self.anomaly_threshold_spin, 0, 1)
        
        # Consensus requirement
        anomaly_layout.addWidget(QLabel("Consensus Required:"), 1, 0)
        self.consensus_spin = QSpinBox()
        self.consensus_spin.setRange(1, 5)
        self.consensus_spin.setValue(self.config.anomaly_detection.consensus_required)
        anomaly_layout.addWidget(self.consensus_spin, 1, 1)
        
        anomaly_group.setLayout(anomaly_layout)
        layout.addWidget(anomaly_group)
        
        # Apply button
        self.apply_button = QPushButton("Apply Configuration")
        self.apply_button.clicked.connect(self.apply_configuration)
        layout.addWidget(self.apply_button)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def apply_configuration(self):
        """Apply configuration changes"""
        self.config.image_processing.noise_reduction_sigma = self.noise_sigma_spin.value()
        self.config.image_processing.edge_detection_threshold = self.detection_threshold_spin.value()
        self.config.image_processing.object_detection_min_area = self.min_area_spin.value()
        self.config.anomaly_detection.detection_threshold = self.anomaly_threshold_spin.value()
        self.config.anomaly_detection.consensus_required = self.consensus_spin.value()


class CosmicAnomalyDetectorGUI(QMainWindow):
    """Main GUI application window"""
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.current_results = None
        self.settings = QSettings("CosmicAnomalyDetector", "GUI")
        
        # Setup logging
        setup_logging(
            log_level=self.config.log_level,
            console_output=True,
            file_output=True
        )
        
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Cosmic Anomaly Detector - JWST Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Image analysis tab
        self.image_tab = self.create_image_analysis_tab()
        self.tab_widget.addTab(self.image_tab, "Image Analysis")
        
        # Object analysis tab
        self.object_tab = ObjectAnalysisWidget()
        self.tab_widget.addTab(self.object_tab, "Object Analysis")
        
        # Configuration tab
        self.config_tab = ConfigurationWidget(self.config)
        self.tab_widget.addTab(self.config_tab, "Configuration")
        
        # Progress bar and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open FITS Image...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save Results...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')
        
        analyze_action = QAction('Analyze Current Image', self)
        analyze_action.setShortcut('F5')
        analyze_action.triggered.connect(self.analyze_image)
        analysis_menu.addAction(analyze_action)
        
        batch_action = QAction('Batch Analysis...', self)
        batch_action.triggered.connect(self.batch_analysis)
        analysis_menu.addAction(batch_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_toolbar(self):
        """Create application toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Open file button
        open_btn = QPushButton("Open FITS")
        open_btn.clicked.connect(self.open_file)
        toolbar.addWidget(open_btn)
        
        toolbar.addSeparator()
        
        # Analyze button
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        toolbar.addWidget(self.analyze_btn)
        
        toolbar.addSeparator()
        
        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.save_results)
        self.export_btn.setEnabled(False)
        toolbar.addWidget(self.export_btn)
        
    def create_image_analysis_tab(self):
        """Create the image analysis tab"""
        widget = QWidget()
        layout = QHBoxLayout()
        
        # Left side - image display
        self.image_display = ImageDisplayWidget()
        layout.addWidget(self.image_display, 2)
        
        # Right side - log and info
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Info panel
        info_group = QGroupBox("Image Information")
        info_layout = QGridLayout()
        
        self.image_path_label = QLabel("No image loaded")
        self.image_size_label = QLabel("Size: N/A")
        self.image_type_label = QLabel("Type: N/A")
        
        info_layout.addWidget(QLabel("Path:"), 0, 0)
        info_layout.addWidget(self.image_path_label, 0, 1)
        info_layout.addWidget(self.image_size_label, 1, 0, 1, 2)
        info_layout.addWidget(self.image_type_label, 2, 0, 1, 2)
        
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        # Log display
        log_group = QGroupBox("Analysis Log")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(200)
        log_layout.addWidget(self.log_display)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        right_widget.setLayout(right_layout)
        layout.addWidget(right_widget, 1)
        
        widget.setLayout(layout)
        return widget
        
    def open_file(self):
        """Open FITS file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open FITS Image", "", 
            "FITS files (*.fits *.fit *.fts);;All files (*)"
        )
        
        if file_path:
            self.load_image(file_path)
            
    def load_image(self, file_path: str):
        """Load FITS image"""
        try:
            from astropy.io import fits
            
            with fits.open(file_path) as hdul:
                image_data = hdul[0].data
                header = hdul[0].header
                
            # Update image display
            self.image_display.set_image(image_data)
            
            # Update info labels
            self.image_path_label.setText(Path(file_path).name)
            self.image_size_label.setText(f"Size: {image_data.shape}")
            self.image_type_label.setText(f"Type: {image_data.dtype}")
            
            # Enable analyze button
            self.analyze_btn.setEnabled(True)
            
            # Store current file path
            self.current_file_path = file_path
            
            self.log_message(f"Loaded image: {file_path}")
            self.status_bar.showMessage(f"Loaded: {Path(file_path).name}")
            
        except Exception as e:
            self.show_error(f"Failed to load image: {str(e)}")
            
    def analyze_image(self):
        """Start image analysis"""
        if not hasattr(self, 'current_file_path'):
            self.show_error("No image loaded")
            return
            
        # Disable controls during analysis
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start analysis worker
        self.analysis_worker = ImageAnalysisWorker(self.current_file_path, self.config)
        self.analysis_worker.progress_updated.connect(self.progress_bar.setValue)
        self.analysis_worker.status_updated.connect(self.log_message)
        self.analysis_worker.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        self.analysis_worker.start()
        
    def on_analysis_complete(self, results: Dict):
        """Handle completed analysis"""
        self.current_results = results
        
        # Update image display with processed image
        self.image_display.set_image(results['processed_image'])
        self.image_display.set_objects(results['objects'])
        
        # Update object analysis tab
        self.object_tab.update_objects(results['objects'], results['geometric_features'])
        
        # Log completion
        num_objects = len(results['objects'])
        artificial_candidates = sum(1 for f in results['geometric_features'] 
                                  if f.artificial_probability > 0.8)
        
        self.log_message(f"Analysis complete! Found {num_objects} objects, "
                        f"{artificial_candidates} artificial candidates")
        
        # Re-enable controls
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Analysis complete")
        
    def on_analysis_error(self, error_msg: str):
        """Handle analysis error"""
        self.show_error(error_msg)
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Analysis failed")
        
    def save_results(self):
        """Save analysis results"""
        if not self.current_results:
            self.show_error("No results to save")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", 
            "JSON files (*.json);;CSV files (*.csv);;All files (*)"
        )
        
        if file_path:
            try:
                import json

                # Prepare results for serialization
                export_data = {
                    'metadata': {
                        'source_file': getattr(self, 'current_file_path', 'unknown'),
                        'analysis_timestamp': str(pd.Timestamp.now()),
                        'num_objects': len(self.current_results['objects'])
                    },
                    'objects': [],
                    'feature_patterns': self.current_results['feature_patterns']
                }
                
                # Convert objects to serializable format
                for obj, features in zip(self.current_results['objects'], 
                                       self.current_results['geometric_features']):
                    obj_data = {
                        'centroid': obj.centroid,
                        'bbox': obj.bbox,
                        'area': obj.area,
                        'regularity_score': obj.regularity_score,
                        'artificial_probability': features.artificial_probability,
                        'coordinates': {
                            'ra': obj.coordinates.ra.deg if obj.coordinates else None,
                            'dec': obj.coordinates.dec.deg if obj.coordinates else None
                        }
                    }
                    export_data['objects'].append(obj_data)
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
                self.log_message(f"Results saved to: {file_path}")
                self.status_bar.showMessage(f"Results saved")
                
            except Exception as e:
                self.show_error(f"Failed to save results: {str(e)}")
                
    def batch_analysis(self):
        """Start batch analysis of multiple files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select FITS Images for Batch Analysis", "",
            "FITS files (*.fits *.fit *.fts);;All files (*)"
        )
        
        if file_paths:
            # TODO: Implement batch analysis worker
            self.show_info(f"Batch analysis of {len(file_paths)} files "
                          "will be implemented in future version")
            
    def log_message(self, message: str):
        """Add message to log display"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")
        
    def show_error(self, message: str):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        logger.error(message)
        
    def show_info(self, message: str):
        """Show info message"""
        QMessageBox.information(self, "Information", message)
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h3>Cosmic Anomaly Detector</h3>
        <p>AI-powered analysis of James Webb Space Telescope images for detecting artificial structures and anomalies in space.</p>
        
        <p><b>Features:</b></p>
        <ul>
        <li>Advanced noise reduction algorithms</li>
        <li>Object detection and segmentation</li>
        <li>Geometric analysis for artificial structures</li>
        <li>Scale-invariant feature detection</li>
        <li>Physics-based anomaly scoring</li>
        </ul>
        
        <p><b>Version:</b> 0.1.0</p>
        <p><b>Author:</b> Cosmic Anomaly Detection Team</p>
        """
        QMessageBox.about(self, "About Cosmic Anomaly Detector", about_text)
        
    def load_settings(self):
        """Load GUI settings"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
            
    def save_settings(self):
        """Save GUI settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        
    def closeEvent(self, event):
        """Handle application close"""
        self.save_settings()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Cosmic Anomaly Detector")
    app.setApplicationVersion("0.1.0")
    
    # Set application icon (if available)
    try:
        app.setWindowIcon(QIcon("assets/icon.png"))
    except:
        pass
    
    # Create and show main window
    window = CosmicAnomalyDetectorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
