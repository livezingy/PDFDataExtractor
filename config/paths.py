# config/paths.py
"""Output path configuration"""

OUTPUT_STRUCTURE = {
    'data': {
        'description': 'Extracted table data',
        'extensions': ['.csv', '.json']
    },
    'debug': {
        'description': 'Debug information',
        'extensions': ['.png', '.log']
    },
    'preview': {
        'description': 'Preview images',
        'extensions': ['.png']
    }
}

def get_output_structure():
    """Get output folder structure configuration
    
    Returns:
        Dictionary containing output folder structure
    """
    return OUTPUT_STRUCTURE

def get_valid_extensions(subfolder: str) -> list:
    """Get valid file extensions for specific subfolder
    
    Args:
        subfolder: Subfolder name (data/debug/preview)
        
    Returns:
        List of valid file extensions
    """
    if subfolder not in OUTPUT_STRUCTURE:
        raise ValueError(f"Invalid subfolder: {subfolder}")
    return OUTPUT_STRUCTURE[subfolder]['extensions'] 