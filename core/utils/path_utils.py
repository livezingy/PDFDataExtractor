# core/utils/path_utils.py
import os
import sys
import platform
from typing import Dict
from config.paths import get_output_structure, get_valid_extensions

def get_app_dir() -> str:
    """Get application root directory (compatible with packaged mode)
    
    Returns:
        Application root directory path
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def get_tesseract_bin() -> str:
    """Get Tesseract executable path
    
    Returns:
        Tesseract executable path
    """
    if sys.platform == 'win32':
        return os.path.join(get_app_dir(), 'tesseract', 'tesseract.exe')
    return 'tesseract'

def get_tessdata_dir() -> str:
    """Get language pack directory path
    
    Returns:
        Language pack directory path
    """
    if sys.platform == 'win32':
        return os.path.join(get_app_dir(), 'tesseract', 'tessdata')
    return '/usr/share/tesseract-ocr/4.00/tessdata'

def get_output_paths(base_path: str) -> Dict[str, str]:
    """Get all output paths
    
    Args:
        base_path: User specified base output path
        
    Returns:
        Dictionary containing all output paths
    """
    # Get output structure from configuration
    output_structure = get_output_structure()
    
    # Create paths dictionary
    paths = {
        subfolder: os.path.join(base_path, subfolder)
        for subfolder in output_structure.keys()
    }
    
    # Create directories if they don't exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    return paths

def get_output_subpath(params: dict, subfolder: str, filename: str = "") -> str:
    """
    Get output file or directory path in a specific subfolder, with pdf_stem subdir.
    
    Args:
        params: dict containing 'output_path' and 'current_filepath' or 'current_file'
        subfolder: subfolder name (data/debug/preview)
        filename: output filename (optional)
        
    Returns:
        Full output file path (if filename is given) or directory path (if filename is empty)
        
    Raises:
        ValueError: If required arguments are missing or invalid
    """
    output_structure = get_output_structure()
    output_path = params.get('output_path', '')
    file_path = params.get('current_filepath')
    if not output_path or not file_path:
        raise ValueError("params must contain output_path and current_filepath")
    pdf_stem = os.path.splitext(os.path.basename(file_path))[0]
    if subfolder not in output_structure:
        raise ValueError(f"Invalid subfolder: {subfolder}")
    full_dir = os.path.join(output_path, subfolder, pdf_stem)
    os.makedirs(full_dir, exist_ok=True)
    if filename:
        return os.path.join(full_dir, filename)
    else:
        return full_dir