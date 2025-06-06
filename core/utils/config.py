# core/utils/config.py
import json
import os
from typing import Any, Dict, Optional

class Config:
    """Configuration manager
    
    Handles loading, saving and accessing configuration settings.
    Supports JSON format configuration files.
    """
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize configuration manager
        
        Args:
            config_file: Configuration file path
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.load()
        
    def load(self) -> None:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Failed to load configuration: {e}")
                self.config = {}
                
    def save(self) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save configuration: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        self.save()
        
    def update(self, config: Dict[str, Any]) -> None:
        """Update multiple configuration values
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        self.save()
        
    def remove(self, key: str) -> None:
        """Remove configuration value
        
        Args:
            key: Configuration key
        """
        if key in self.config:
            del self.config[key]
            self.save()
            
    def clear(self) -> None:
        """Clear all configuration values"""
        self.config.clear()
        self.save()
        
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy() 