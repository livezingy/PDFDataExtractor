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

def get_output_paths(base_path: str, pdf_stem: str = None) -> Dict[str, str]:
    """Get all output paths, optionally with a subfolder for each PDF
    
    Args:
        base_path: User specified base output path
        pdf_stem: PDF file name without extension (optional)
        
    Returns:
        Dictionary containing all output paths
    """
    # Get output structure from configuration
    output_structure = get_output_structure()
    
    if pdf_stem:
        # Create paths with PDF stem subfolder
        paths = {
            subfolder: os.path.join(base_path, subfolder, pdf_stem)
            for subfolder in output_structure.keys()
        }
    else:
        # Create paths without PDF stem subfolder
        paths = {
            subfolder: os.path.join(base_path, subfolder)
            for subfolder in output_structure.keys()
        }
    
    # Create directories if they don't exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    return paths

def get_output_subpath(base_path: str, subfolder: str, filename: str, pdf_stem: str) -> str:
    """Get output file path in specific subfolder, with pdf_stem subdir
    
    Args:
        base_path: Base output path
        subfolder: Subfolder name (data/debug/preview)
        filename: Output filename
        pdf_stem: PDF file name without extension (required)
        
    Returns:
        Full output file path
        
    Raises:
        ValueError: If subfolder is invalid or file extension is not allowed
    """
    output_structure = get_output_structure()
    if subfolder not in output_structure:
        raise ValueError(f"Invalid subfolder: {subfolder}")
    file_ext = os.path.splitext(filename)[1].lower()
    """  valid_extensions = get_valid_extensions(subfolder)
    if file_ext not in valid_extensions:
        raise ValueError(
            f"Invalid file extension {file_ext} for {subfolder}. "
            f"Allowed extensions: {valid_extensions}"
        ) """
    # generate path：output/subfolder/pdf_stem/filename
    full_dir = os.path.join(base_path, subfolder, pdf_stem)
    os.makedirs(full_dir, exist_ok=True)
    return os.path.join(full_dir, filename)