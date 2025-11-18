from .nodes import ImageAlphaCrop, ImageAlphaCropAdvanced

NODE_CLASS_MAPPINGS = {
    "ImageAlphaCrop": ImageAlphaCrop,
    "ImageAlphaCropAdvanced": ImageAlphaCropAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAlphaCrop": "Image Alpha Crop",
    "ImageAlphaCropAdvanced": "Image Alpha Crop (Advanced)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']