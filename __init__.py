"""
Background Remover node using RMBG-2.0 for ComfyUI
Removes backgrounds from images using the RMBG-2.0 model from briaai
"""

from .bgrem import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']