"""Calibration plugin for NaviGraph unified architecture.

Loads camera calibration transformation matrix and provides it as a shared resource.
"""

import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd

from ...core.navigraph_plugin import NaviGraphPlugin
from ...core.exceptions import NavigraphError
from ...core.registry import register_data_source_plugin


@register_data_source_plugin("calibration")
class CalibrationPlugin(NaviGraphPlugin):
    """Provides calibration transformation matrix as a shared resource."""
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.config.get('name', 'calibration'),
            'type': 'calibration',
            'description': 'Loads camera calibration transformation matrix',
            'provides': ['calibration_matrix'],
            'augments': []
        }
    
    def provide(self, shared_resources: Dict[str, Any]) -> None:
        """Load calibration matrix and add to shared resources.
        
        Args:
            shared_resources: Dictionary to store calibration matrix
        """
        # Validate we have discovered files
        if not self.discovered_files:
            raise NavigraphError(
                f"CalibrationPlugin requires transform_matrix.npy file but none found. "
                f"Check file_pattern in config: {self.config.get('file_pattern')}"
            )
        
        calibration_file = self.discovered_files[0]  # Use first discovered file
        self.logger.info(f"Loading calibration from: {calibration_file.name}")
        
        try:
            # Load transformation matrix
            transformation_matrix = np.load(calibration_file)
            
            # Validate matrix shape (should be 3x3 homography matrix)
            if transformation_matrix.shape != (3, 3):
                raise NavigraphError(
                    f"Invalid transformation matrix shape: {transformation_matrix.shape}. "
                    f"Expected (3, 3) homography matrix."
                )
            
            # Validate matrix is valid (determinant should not be zero)
            det = np.linalg.det(transformation_matrix)
            if np.abs(det) < 1e-10:
                raise NavigraphError(
                    f"Transformation matrix is singular (determinant={det:.3e}). "
                    f"This indicates an invalid calibration."
                )
            
            # Add to shared resources
            shared_resources['calibration_matrix'] = transformation_matrix
            
            self.logger.info(
                f"✓ Calibration matrix loaded: shape={transformation_matrix.shape}, "
                f"determinant={det:.3f}"
            )
            
        except Exception as e:
            raise NavigraphError(
                f"Failed to load calibration from {calibration_file}: {str(e)}"
            ) from e
    
    def augment_data(self, dataframe: pd.DataFrame, shared_resources: Dict[str, Any]) -> pd.DataFrame:
        """Calibration plugin doesn't augment data - only provides resource.
        
        Args:
            dataframe: Current DataFrame
            shared_resources: Available shared resources
            
        Returns:
            DataFrame unchanged
        """
        # Calibration doesn't add columns, just provides the matrix
        return dataframe