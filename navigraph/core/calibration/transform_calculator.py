"""Transformation matrix calculation for camera-to-map calibration."""

from typing import Tuple, Optional, NamedTuple
from enum import Enum
from pathlib import Path
import numpy as np
import cv2
from loguru import logger


class TransformMethod(Enum):
    """Supported transformation methods."""
    AFFINE = "affine"
    HOMOGRAPHY = "homography"
    HOMOGRAPHY_RANSAC = "homography_ransac"


class CalibrationResult(NamedTuple):
    """Result of calibration with quality metrics."""
    transform_matrix: np.ndarray
    method: TransformMethod
    reprojection_error: float
    source_points: np.ndarray
    target_points: np.ndarray
    
    def save(self, output_path: Path) -> None:
        """Save transformation matrix to file.
        
        Args:
            output_path: Path where to save the transform matrix (.npy file)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, self.transform_matrix)
        logger.info(f"Transform matrix saved to: {output_path}")
        
        # Save metadata alongside matrix
        metadata_path = output_path.with_suffix('.metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Calibration Method: {self.method.value}\n")
            f.write(f"Reprojection Error: {self.reprojection_error:.3f} pixels\n")
            f.write(f"Number of Points: {len(self.source_points)}\n")
            f.write(f"Transform Matrix Shape: {self.transform_matrix.shape}\n")
        
        logger.info(f"Metadata saved to: {metadata_path}")


class TransformCalculator:
    """Calculate transformation matrices between coordinate systems."""
    
    @staticmethod
    def calculate_transform(
        source_points: np.ndarray,
        target_points: np.ndarray,
        method: TransformMethod = TransformMethod.HOMOGRAPHY_RANSAC
    ) -> CalibrationResult:
        """Calculate transformation matrix with quality metrics.
        
        Args:
            source_points: Points from source image (camera) [N, 2]
            target_points: Corresponding points from target image (map) [N, 2]
            method: Transformation method to use
            
        Returns:
            CalibrationResult with matrix and quality metrics
            
        Raises:
            ValueError: If insufficient points or calculation fails
        """
        if len(source_points) != len(target_points):
            raise ValueError("Source and target points must have same length")
            
        if len(source_points) < 4:
            raise ValueError("At least 4 point pairs required for calibration")
        
        # Ensure float32 for OpenCV
        src_pts = source_points.astype(np.float32)
        dst_pts = target_points.astype(np.float32)
        
        # Calculate transformation based on method
        if method == TransformMethod.AFFINE:
            if len(source_points) < 3:
                raise ValueError("Affine transformation requires at least 3 points")
            transform_matrix = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
            
        elif method == TransformMethod.HOMOGRAPHY:
            transform_matrix, _ = cv2.findHomography(
                src_pts, dst_pts, method=0
            )
            
        elif method == TransformMethod.HOMOGRAPHY_RANSAC:
            transform_matrix, _ = cv2.findHomography(
                src_pts, dst_pts, 
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
        else:
            raise ValueError(f"Unsupported transformation method: {method}")
        
        if transform_matrix is None:
            raise ValueError("Failed to calculate transformation matrix")
        
        # Calculate reprojection error
        reprojection_error = TransformCalculator._calculate_reprojection_error(
            src_pts, dst_pts, transform_matrix, method
        )
        
        logger.info(f"Calculated {method.value} transformation")
        logger.info(f"Reprojection error: {reprojection_error:.2f} pixels")
        
        return CalibrationResult(
            transform_matrix=transform_matrix,
            method=method,
            reprojection_error=reprojection_error,
            source_points=source_points,
            target_points=target_points
        )
    
    @staticmethod
    def _calculate_reprojection_error(
        source_points: np.ndarray,
        target_points: np.ndarray,
        transform_matrix: np.ndarray,
        method: TransformMethod
    ) -> float:
        """Calculate mean reprojection error."""
        if method == TransformMethod.AFFINE:
            # For affine, add homogeneous coordinate
            src_homogeneous = np.column_stack([source_points, np.ones(len(source_points))])
            projected_points = (transform_matrix @ src_homogeneous.T).T
        else:
            # For homography
            projected_points = cv2.perspectiveTransform(
                source_points.reshape(-1, 1, 2), transform_matrix
            ).reshape(-1, 2)
        
        # Calculate Euclidean distances
        distances = np.sqrt(np.sum((projected_points - target_points) ** 2, axis=1))
        return float(np.mean(distances))
    
    @staticmethod
    def test_transform(
        transform_matrix: np.ndarray,
        test_points: np.ndarray,
        method: TransformMethod = TransformMethod.HOMOGRAPHY_RANSAC
    ) -> np.ndarray:
        """Test transformation on new points.
        
        Args:
            transform_matrix: Previously calculated transformation matrix
            test_points: Points to transform [N, 2]
            method: Method used to calculate the matrix
            
        Returns:
            Transformed points [N, 2]
        """
        test_pts = test_points.astype(np.float32)
        
        if method == TransformMethod.AFFINE:
            # For affine transformation
            homogeneous = np.column_stack([test_pts, np.ones(len(test_pts))])
            transformed = (transform_matrix @ homogeneous.T).T
            return transformed[:, :2]
        else:
            # For homography
            return cv2.perspectiveTransform(
                test_pts.reshape(-1, 1, 2), transform_matrix
            ).reshape(-1, 2)