"""
Advanced Computer Vision Components for Cosmic Anomaly Detection

Implements sophisticated noise reduction, object detection, and geometric analysis
for identifying artificial structures in astronomical images.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import astropy.units as u
import cv2
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage, signal
from skimage import filters, measure, morphology, segmentation
from skimage.feature import corner_harris, corner_peaks, peak_local_maxima

from ..utils.config import get_config
from ..utils.logging import get_logger, log_processing_stage

logger = get_logger(__name__)


@dataclass
class DetectedObject:
    """Container for detected astronomical objects"""
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    area: float
    perimeter: float
    eccentricity: float
    major_axis_length: float
    minor_axis_length: float
    orientation: float
    intensity_mean: float
    intensity_max: float
    regularity_score: float
    coordinates: Optional[SkyCoord] = None
    confidence: float = 0.0


@dataclass
class GeometricFeatures:
    """Container for geometric analysis features"""
    regularity_score: float
    symmetry_score: float
    edge_sharpness: float
    circularity: float
    rectangularity: float
    artificial_probability: float


class AdvancedNoiseReducer:
    """Advanced noise reduction algorithms for space telescope images"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config().image_processing
        
    def adaptive_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive bilateral filtering for noise reduction"""
        logger.debug("Applying adaptive bilateral filter")
        
        # Estimate noise level
        noise_std = np.std(image - cv2.medianBlur(image.astype(np.float32), 5))
        
        # Adaptive parameters based on noise level
        d = int(max(5, min(15, noise_std * 10)))
        sigma_color = max(10, noise_std * 20)
        sigma_space = max(10, noise_std * 20)
        
        # Convert to appropriate type for OpenCV
        if image.dtype != np.uint8:
            image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image_uint8 = image_norm.astype(np.uint8)
        else:
            image_uint8 = image
            
        filtered = cv2.bilateralFilter(image_uint8, d, sigma_color, sigma_space)
        
        # Convert back to original range
        if image.dtype != np.uint8:
            filtered = filtered.astype(image.dtype) * (image.max() / 255.0)
            
        return filtered
    
    def non_local_means_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising"""
        logger.debug("Applying non-local means denoising")
        
        if image.dtype != np.uint8:
            image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image_uint8 = image_norm.astype(np.uint8)
        else:
            image_uint8 = image
            
        denoised = cv2.fastNlMeansDenoising(image_uint8, None, 10, 7, 21)
        
        if image.dtype != np.uint8:
            denoised = denoised.astype(image.dtype) * (image.max() / 255.0)
            
        return denoised
    
    def wavelet_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply wavelet-based denoising"""
        logger.debug("Applying wavelet denoising")
        
        try:
            import pywt

            # Decompose image using wavelets
            coeffs = pywt.wavedec2(image, 'db4', levels=3)
            
            # Estimate noise threshold
            sigma = np.std(coeffs[-1])
            threshold = sigma * np.sqrt(2 * np.log(image.size))
            
            # Apply soft thresholding
            coeffs_thresh = list(coeffs)
            for i in range(1, len(coeffs)):
                coeffs_thresh[i] = tuple([
                    pywt.threshold(detail, threshold, 'soft') 
                    for detail in coeffs[i]
                ])
            
            # Reconstruct image
            denoised = pywt.waverec2(coeffs_thresh, 'db4')
            return denoised.astype(image.dtype)
            
        except ImportError:
            logger.warning("PyWavelets not available, using Gaussian filter")
            return ndimage.gaussian_filter(image, sigma=1.0)


class ObjectDetector:
    """Advanced object detection and segmentation for astronomical images"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config().image_processing
        self.min_area = self.config.get('object_detection_min_area', 100)
        
    def detect_objects(self, image: np.ndarray, 
                      wcs: Optional[WCS] = None) -> List[DetectedObject]:
        """Detect and characterize objects in astronomical image"""
        logger.info("Starting object detection")
        
        # Preprocessing
        processed = self._preprocess_for_detection(image)
        
        # Segmentation
        labels = self._segment_objects(processed)
        
        # Extract object properties
        objects = self._extract_object_properties(labels, image, wcs)
        
        logger.info(f"Detected {len(objects)} objects")
        return objects
    
    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for object detection"""
        # Normalize
        normalized = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        
        # Enhance contrast
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(
            (normalized * 255).astype(np.uint8)
        ) / 255.0
        
        # Edge preservation smoothing
        smoothed = cv2.edgePreservingFilter(
            (enhanced * 255).astype(np.uint8), flags=1, sigma_s=60, sigma_r=0.4
        ) / 255.0
        
        return smoothed
    
    def _segment_objects(self, image: np.ndarray) -> np.ndarray:
        """Segment objects using advanced techniques"""
        # Multi-scale segmentation
        
        # Local threshold
        threshold_local = filters.threshold_local(image, block_size=35, offset=0.01)
        binary_local = image > threshold_local
        
        # Otsu threshold
        threshold_otsu = filters.threshold_otsu(image)
        binary_otsu = image > threshold_otsu
        
        # Combine thresholds
        binary_combined = binary_local | binary_otsu
        
        # Morphological operations
        selem = morphology.disk(2)
        binary_cleaned = morphology.closing(binary_combined, selem)
        binary_cleaned = morphology.remove_small_objects(binary_cleaned, 
                                                        min_size=self.min_area)
        
        # Watershed segmentation for separating touching objects
        distance = ndimage.distance_transform_edt(binary_cleaned)
        local_maxima = peak_local_maxima(distance, min_distance=20, 
                                       threshold_abs=0.3*distance.max(),
                                       indices=False)
        markers = measure.label(local_maxima)
        labels = segmentation.watershed(-distance, markers, mask=binary_cleaned)
        
        return labels
    
    def _extract_object_properties(self, labels: np.ndarray, 
                                 original_image: np.ndarray,
                                 wcs: Optional[WCS] = None) -> List[DetectedObject]:
        """Extract properties of detected objects"""
        objects = []
        
        for region in measure.regionprops(labels, intensity_image=original_image):
            if region.area < self.min_area:
                continue
                
            # Basic properties
            centroid = region.centroid
            bbox = region.bbox
            
            # Geometric properties
            area = region.area
            perimeter = region.perimeter
            eccentricity = region.eccentricity
            major_axis = region.major_axis_length
            minor_axis = region.minor_axis_length
            orientation = region.orientation
            
            # Intensity properties
            intensity_mean = region.mean_intensity
            intensity_max = region.max_intensity
            
            # Calculate regularity score
            regularity_score = self._calculate_regularity(region)
            
            # Convert pixel coordinates to sky coordinates if WCS available
            coordinates = None
            if wcs is not None:
                try:
                    sky_coord = wcs.pixel_to_world(centroid[1], centroid[0])
                    coordinates = sky_coord
                except Exception as e:
                    logger.warning(f"Failed to convert coordinates: {e}")
            
            obj = DetectedObject(
                centroid=centroid,
                bbox=bbox,
                area=area,
                perimeter=perimeter,
                eccentricity=eccentricity,
                major_axis_length=major_axis,
                minor_axis_length=minor_axis,
                orientation=orientation,
                intensity_mean=intensity_mean,
                intensity_max=intensity_max,
                regularity_score=regularity_score,
                coordinates=coordinates
            )
            
            objects.append(obj)
            
        return objects
    
    def _calculate_regularity(self, region) -> float:
        """Calculate geometric regularity score for an object"""
        # Combine multiple regularity metrics
        
        # Circularity (4π × Area / Perimeter²)
        circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
        circularity = min(circularity, 1.0)  # Cap at 1.0
        
        # Aspect ratio regularity
        aspect_ratio = region.minor_axis_length / max(region.major_axis_length, 1e-6)
        
        # Solidity (convex hull area ratio)
        solidity = region.solidity
        
        # Extent (object area / bounding box area)
        extent = region.extent
        
        # Combine metrics
        regularity = (circularity * 0.3 + 
                     aspect_ratio * 0.3 + 
                     solidity * 0.2 + 
                     extent * 0.2)
        
        return regularity


class GeometricAnalyzer:
    """Analyzes geometric properties to identify artificial structures"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config().image_processing
        
    def analyze_geometry(self, objects: List[DetectedObject], 
                        image: np.ndarray) -> List[GeometricFeatures]:
        """Analyze geometric features of detected objects"""
        logger.info(f"Analyzing geometry of {len(objects)} objects")
        
        features = []
        for obj in objects:
            feature = self._analyze_single_object(obj, image)
            features.append(feature)
            
        return features
    
    def _analyze_single_object(self, obj: DetectedObject, 
                              image: np.ndarray) -> GeometricFeatures:
        """Analyze geometric features of a single object"""
        
        # Extract object region
        min_row, min_col, max_row, max_col = obj.bbox
        object_region = image[min_row:max_row, min_col:max_col]
        
        # Calculate various geometric features
        regularity_score = obj.regularity_score
        symmetry_score = self._calculate_symmetry(object_region)
        edge_sharpness = self._calculate_edge_sharpness(object_region)
        circularity = self._calculate_circularity(obj)
        rectangularity = self._calculate_rectangularity(obj)
        
        # Combine features to estimate artificial probability
        artificial_probability = self._estimate_artificial_probability(
            regularity_score, symmetry_score, edge_sharpness, 
            circularity, rectangularity
        )
        
        return GeometricFeatures(
            regularity_score=regularity_score,
            symmetry_score=symmetry_score,
            edge_sharpness=edge_sharpness,
            circularity=circularity,
            rectangularity=rectangularity,
            artificial_probability=artificial_probability
        )
    
    def _calculate_symmetry(self, region: np.ndarray) -> float:
        """Calculate symmetry score of an object region"""
        if region.size == 0:
            return 0.0
            
        # Horizontal symmetry
        left_half = region[:, :region.shape[1]//2]
        right_half = region[:, region.shape[1]//2:]
        right_half_flipped = np.fliplr(right_half)
        
        # Ensure same dimensions
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        if left_half.size > 0 and right_half_flipped.size > 0:
            h_symmetry = 1.0 - np.mean(np.abs(left_half - right_half_flipped))
        else:
            h_symmetry = 0.0
        
        # Vertical symmetry
        top_half = region[:region.shape[0]//2, :]
        bottom_half = region[region.shape[0]//2:, :]
        bottom_half_flipped = np.flipud(bottom_half)
        
        min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half_flipped = bottom_half_flipped[:min_height, :]
        
        if top_half.size > 0 and bottom_half_flipped.size > 0:
            v_symmetry = 1.0 - np.mean(np.abs(top_half - bottom_half_flipped))
        else:
            v_symmetry = 0.0
        
        return max(h_symmetry, v_symmetry)
    
    def _calculate_edge_sharpness(self, region: np.ndarray) -> float:
        """Calculate edge sharpness indicating artificial construction"""
        if region.size == 0:
            return 0.0
            
        # Calculate gradients
        grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Sharp edges have high gradient variance
        edge_variance = np.var(grad_magnitude)
        
        # Normalize by image intensity range
        intensity_range = region.max() - region.min()
        if intensity_range > 0:
            normalized_sharpness = edge_variance / (intensity_range**2)
        else:
            normalized_sharpness = 0.0
            
        return min(normalized_sharpness, 1.0)
    
    def _calculate_circularity(self, obj: DetectedObject) -> float:
        """Calculate circularity metric"""
        if obj.perimeter <= 0:
            return 0.0
        return 4 * np.pi * obj.area / (obj.perimeter ** 2)
    
    def _calculate_rectangularity(self, obj: DetectedObject) -> float:
        """Calculate how rectangular an object is"""
        # Calculate bounding box area
        bbox_area = ((obj.bbox[2] - obj.bbox[0]) * 
                    (obj.bbox[3] - obj.bbox[1]))
        
        if bbox_area <= 0:
            return 0.0
            
        return obj.area / bbox_area
    
    def _estimate_artificial_probability(self, regularity: float, 
                                       symmetry: float, edge_sharpness: float,
                                       circularity: float, 
                                       rectangularity: float) -> float:
        """Estimate probability that object is artificial"""
        
        # Weights for different features (based on what's unusual in nature)
        weights = {
            'high_regularity': 0.3,
            'high_symmetry': 0.25,
            'sharp_edges': 0.2,
            'perfect_geometry': 0.25
        }
        
        # High regularity score
        high_regularity = regularity ** 2
        
        # High symmetry score
        high_symmetry = symmetry ** 2
        
        # Sharp artificial edges
        sharp_edges = edge_sharpness
        
        # Perfect geometric shapes (circles or rectangles)
        perfect_circle = circularity if circularity > 0.9 else 0.0
        perfect_rectangle = rectangularity if rectangularity > 0.9 else 0.0
        perfect_geometry = max(perfect_circle, perfect_rectangle)
        
        # Combine features
        artificial_score = (
            weights['high_regularity'] * high_regularity +
            weights['high_symmetry'] * high_symmetry +
            weights['sharp_edges'] * sharp_edges +
            weights['perfect_geometry'] * perfect_geometry
        )
        
        return min(artificial_score, 1.0)


class ScaleInvariantFeatureDetector:
    """Detects scale-invariant features for structure analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config().image_processing
        
    def detect_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect scale-invariant keypoints and descriptors"""
        logger.debug("Detecting scale-invariant features")
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            image_uint8 = image
            
        # Use ORB as a free alternative to SIFT
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(image_uint8, None)
        
        # Convert keypoints to array format
        if keypoints:
            keypoint_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        else:
            keypoint_array = np.array([])
            
        return keypoint_array, descriptors
    
    def analyze_feature_patterns(self, keypoints: np.ndarray, 
                               descriptors: np.ndarray) -> Dict[str, float]:
        """Analyze patterns in detected features"""
        if len(keypoints) == 0:
            return {'regularity': 0.0, 'clustering': 0.0, 'alignment': 0.0}
        
        # Feature distribution regularity
        regularity = self._calculate_feature_regularity(keypoints)
        
        # Feature clustering
        clustering = self._calculate_feature_clustering(keypoints)
        
        # Feature alignment (artificial structures often have aligned features)
        alignment = self._calculate_feature_alignment(keypoints)
        
        return {
            'regularity': regularity,
            'clustering': clustering,
            'alignment': alignment
        }
    
    def _calculate_feature_regularity(self, keypoints: np.ndarray) -> float:
        """Calculate regularity of feature point distribution"""
        if len(keypoints) < 3:
            return 0.0
            
        # Calculate distances between consecutive points
        distances = []
        for i in range(len(keypoints) - 1):
            dist = np.linalg.norm(keypoints[i+1] - keypoints[i])
            distances.append(dist)
            
        if len(distances) == 0:
            return 0.0
            
        # Regular patterns have low distance variance
        distance_var = np.var(distances)
        distance_mean = np.mean(distances)
        
        if distance_mean > 0:
            regularity = 1.0 / (1.0 + distance_var / distance_mean)
        else:
            regularity = 0.0
            
        return regularity
    
    def _calculate_feature_clustering(self, keypoints: np.ndarray) -> float:
        """Calculate feature clustering coefficient"""
        if len(keypoints) < 3:
            return 0.0
            
        # Use simple clustering metric based on nearest neighbor distances
        total_clustering = 0.0
        
        for i, point in enumerate(keypoints):
            # Find distances to all other points
            distances = [np.linalg.norm(point - other) 
                        for j, other in enumerate(keypoints) if i != j]
            
            if len(distances) >= 2:
                distances.sort()
                # Clustering based on ratio of nearest to second nearest
                if distances[1] > 0:
                    clustering = distances[0] / distances[1]
                    total_clustering += clustering
                    
        if len(keypoints) > 0:
            return total_clustering / len(keypoints)
        else:
            return 0.0
    
    def _calculate_feature_alignment(self, keypoints: np.ndarray) -> float:
        """Calculate feature alignment score"""
        if len(keypoints) < 3:
            return 0.0
            
        # Check for linear alignments
        max_alignment = 0.0
        
        for i in range(len(keypoints)):
            for j in range(i+1, len(keypoints)):
                # Vector between two points
                direction = keypoints[j] - keypoints[i]
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm == 0:
                    continue
                    
                direction = direction / direction_norm
                
                # Count points aligned with this direction
                aligned_count = 0
                for k in range(len(keypoints)):
                    if k == i or k == j:
                        continue
                        
                    # Vector to third point
                    to_point = keypoints[k] - keypoints[i]
                    to_point_norm = np.linalg.norm(to_point)
                    
                    if to_point_norm == 0:
                        continue
                        
                    to_point = to_point / to_point_norm
                    
                    # Check alignment
                    alignment = abs(np.dot(direction, to_point))
                    if alignment > 0.9:  # Highly aligned
                        aligned_count += 1
                        
                # Calculate alignment score
                if len(keypoints) > 2:
                    alignment_score = aligned_count / (len(keypoints) - 2)
                    max_alignment = max(max_alignment, alignment_score)
                    
        return max_alignment
