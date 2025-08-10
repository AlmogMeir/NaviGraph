"""GUI utilities for handling matplotlib backends and fallback options.

This module provides utilities to ensure consistent GUI behavior across different
environments, with automatic fallback to alternative backends when needed.
"""

import os
import sys
import warnings
from typing import Optional, Tuple, Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class BackendManager:
    """Manages matplotlib backend configuration and GUI availability."""
    
    def __init__(self):
        self.gui_available = False
        self.current_backend = None
        self._setup_backend()
    
    def _setup_backend(self):
        """Setup matplotlib backend with fallback options."""
        # First, try to handle display issues
        if not os.environ.get('DISPLAY'):
            # No display available, go straight to Agg
            matplotlib.use('Agg', force=True)
            self.gui_available = False
            self.current_backend = 'Agg'
            return
            
        # Fix QT_API environment variable if needed
        if os.environ.get('QT_API') == 'pyqt':
            os.environ['QT_API'] = 'pyqt5'
        
        # List of backends to try in order of preference
        # Put TkAgg first as it's more reliable in headless environments
        gui_backends = ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'Qt4Agg']
        
        for backend in gui_backends:
            try:
                # Suppress all warnings and stdout/stderr during testing
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Redirect stderr temporarily
                    original_stderr = sys.stderr
                    sys.stderr = open(os.devnull, 'w')
                    
                    matplotlib.use(backend, force=True)
                    
                    # Test if backend actually works
                    fig = plt.figure()
                    plt.close(fig)
                    
                    # Restore stderr
                    sys.stderr.close()
                    sys.stderr = original_stderr
                
                self.gui_available = True
                self.current_backend = backend
                return
                
            except Exception:
                # Restore stderr if something went wrong
                if 'original_stderr' in locals() and sys.stderr != original_stderr:
                    sys.stderr.close()
                    sys.stderr = original_stderr
                continue
        
        # If all GUI backends fail, fall back to Agg
        matplotlib.use('Agg', force=True)
        self.gui_available = False
        self.current_backend = 'Agg'
    
    def is_gui_available(self) -> bool:
        """Check if GUI functionality is available."""
        return self.gui_available
    
    def get_current_backend(self) -> str:
        """Get the currently active backend."""
        return self.current_backend or matplotlib.get_backend()
    
    def create_figure(self, figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
        """Create a matplotlib figure with proper backend handling."""
        if not self.gui_available:
            # For non-GUI backends, we can still create figures for saving
            return plt.figure(figsize=figsize)
        
        try:
            return plt.figure(figsize=figsize)
        except Exception as e:
            warnings.warn(f"Failed to create GUI figure: {e}. Falling back to non-GUI mode.")
            matplotlib.use('Agg', force=True)
            self.gui_available = False
            self.current_backend = 'Agg'
            return plt.figure(figsize=figsize)


# Global backend manager instance
_backend_manager = None

def get_backend_manager() -> BackendManager:
    """Get the global backend manager instance."""
    global _backend_manager
    if _backend_manager is None:
        _backend_manager = BackendManager()
    return _backend_manager


def is_gui_available() -> bool:
    """Check if GUI functionality is available."""
    return get_backend_manager().is_gui_available()


def get_current_backend() -> str:
    """Get the currently active matplotlib backend."""
    return get_backend_manager().get_current_backend()


def create_figure(figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
    """Create a matplotlib figure with proper backend handling."""
    return get_backend_manager().create_figure(figsize)


def safe_show(fig: plt.Figure, block: bool = True) -> bool:
    """Safely show a matplotlib figure with fallback handling.
    
    Args:
        fig: Matplotlib figure to show
        block: Whether to block execution (GUI mode only)
        
    Returns:
        True if GUI display was successful, False if fallback used
    """
    if not is_gui_available():
        print("⚠️  GUI not available. Figure cannot be displayed interactively.")
        return False
    
    try:
        plt.show(block=block)
        return True
    except Exception as e:
        print(f"⚠️  Failed to display figure: {e}")
        return False


class OpenCVFallback:
    """OpenCV-based fallback for interactive functionality."""
    
    @staticmethod
    def show_image(image: np.ndarray, title: str = "Image", 
                   wait_key: bool = True) -> Optional[int]:
        """Show image using OpenCV with optional key wait.
        
        Args:
            image: Image array to display
            title: Window title
            wait_key: Whether to wait for key press
            
        Returns:
            Key pressed (if wait_key=True), None otherwise
        """
        try:
            import cv2
            cv2.imshow(title, image)
            if wait_key:
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                return key
            return None
        except ImportError:
            print("⚠️  OpenCV not available for image display")
            return None
        except Exception as e:
            print(f"⚠️  Failed to display image with OpenCV: {e}")
            return None
    
    @staticmethod
    def create_interactive_window(image: np.ndarray, title: str = "Interactive Window",
                                 mouse_callback: Optional[Any] = None) -> bool:
        """Create interactive OpenCV window with mouse callbacks.
        
        Args:
            image: Base image to display
            title: Window title  
            mouse_callback: Mouse callback function
            
        Returns:
            True if window created successfully
        """
        try:
            import cv2
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            if mouse_callback:
                cv2.setMouseCallback(title, mouse_callback)
            cv2.imshow(title, image)
            return True
        except ImportError:
            print("⚠️  OpenCV not available for interactive window")
            return False
        except Exception as e:
            print(f"⚠️  Failed to create interactive window: {e}")
            return False


def print_backend_info():
    """Print information about the current backend configuration."""
    manager = get_backend_manager()
    print(f"📊 Matplotlib Backend Information:")
    print(f"   Current backend: {manager.get_current_backend()}")
    print(f"   GUI available: {manager.is_gui_available()}")
    print(f"   QT_API: {os.environ.get('QT_API', 'not set')}")
    
    # Test backend functionality
    try:
        fig = create_figure((6, 4))
        plt.close(fig)
        print(f"   Backend test: ✅ Passed")
    except Exception as e:
        print(f"   Backend test: ❌ Failed ({e})")


def handle_gui_error(operation_name: str, error: Exception) -> None:
    """Handle GUI-related errors with helpful messages.
    
    Args:
        operation_name: Name of the operation that failed
        error: Exception that occurred
    """
    error_msg = f"GUI operation '{operation_name}' failed: {error}"
    
    if "QT_API" in str(error):
        print(f"⚠️  {error_msg}")
        print("💡 This appears to be a Qt configuration issue.")
        print("   The system has automatically configured Qt settings.")
        print("   If problems persist, try: pip install PyQt5")
    elif "backend" in str(error).lower():
        print(f"⚠️  {error_msg}")
        print("💡 This appears to be a matplotlib backend issue.")
        print("   The system will attempt to use alternative backends.")
    else:
        print(f"⚠️  {error_msg}")
        print("💡 Consider using non-interactive mode or check your display configuration.")


# Initialize backend on import
get_backend_manager()