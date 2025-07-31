"""Calibration data source for NaviGraph.

Loads camera calibration transformation matrices for coordinate transformation.
Stores calibration in shared_resources for other plugins to use.
"""

import numpy as np
import os
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd

from ...core.interfaces import IDataSource, Logger
from ...core.exceptions import DataSourceError
from ...core.base_plugin import BasePlugin
from ...core.registry import register_data_source_plugin


class CalibrationData:
    """Simple wrapper for calibration transformation matrix."""
    
    def __init__(self, transformation_matrix: np.ndarray):
        self.transformation_matrix = transformation_matrix
    
    def get_transformation_matrix(self) -> np.ndarray:
        """Get the transformation matrix."""
        return self.transformation_matrix


@register_data_source_plugin("calibration")
class CalibrationDataSource(BasePlugin, IDataSource):
    """Loads calibration transformation matrix and stores in shared resources.
    
    This plugin loads camera calibration data (transform_matrix.npy) and makes it
    available to other plugins via shared_resources. It does not add any columns
    to the DataFrame - it only provides the calibration resource.
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance: Logger = None):
        """Factory method to create calibration data source from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate calibration configuration."""
        # No required config keys
        pass
    
    def get_provided_column_names(self) -> List[str]:
        """Return column names this data source provides."""
        return []  # Calibration doesn't add DataFrame columns
    
    def integrate_data_into_session(
        self,
        current_dataframe: pd.DataFrame,
        session_config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        logger
    ) -> pd.DataFrame:
        """Load calibration and store in shared resources.
        
        Args:
            current_dataframe: Current DataFrame state
            session_config: Configuration for this data source
            shared_resources: Shared resources dictionary
            logger: Logger instance
            
        Returns:
            DataFrame unchanged (calibration doesn't add columns)
            
        Raises:
            DataSourceError: If calibration file cannot be loaded
        """
        logger.info("Loading calibration transformation matrix")
        
        try:
            # Get the calibration file path from discovered files
            calibration_file = session_config.get('discovered_file_path')
            if not calibration_file:
                raise DataSourceError("No calibration file discovered")
            
            # Load transformation matrix
            if not os.path.exists(calibration_file):
                raise DataSourceError(f"Calibration file not found: {calibration_file}")
            
            transformation_matrix = np.load(calibration_file)
            
            # Validate matrix shape
            if transformation_matrix.shape != (3, 3):
                raise DataSourceError(
                    f"Invalid transformation matrix shape: {transformation_matrix.shape}. Expected (3, 3)"
                )
            
            # Store in shared resources for other plugins to use
            calibration_data = CalibrationData(transformation_matrix)
            shared_resources['calibration'] = calibration_data
            
            logger.info(f"✓ Calibration loaded from: {Path(calibration_file).name}")
            
            # Return DataFrame unchanged - calibration doesn't add columns
            return current_dataframe
            
        except Exception as e:
            raise DataSourceError(f"Failed to load calibration: {str(e)}") from e