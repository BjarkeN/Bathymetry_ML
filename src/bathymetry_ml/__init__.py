"""Bathymetry ML package."""

from pathlib import Path

# Project root is the parent of the src directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def resolve_path(path_str: str) -> Path:
    """Resolve paths relative to project root.
    
    Args:
        path_str: Path string (absolute or relative to project root)
        
    Returns:
        Absolute Path object
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


__all__ = ["PROJECT_ROOT", "resolve_path"]
