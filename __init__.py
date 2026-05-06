from .dreamlite_node import DreamLiteGenerateEdit, DreamLiteClearCache

NODE_CLASS_MAPPINGS = {
    "DreamLiteGenerateEdit": DreamLiteGenerateEdit,
    "DreamLiteClearCache": DreamLiteClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamLiteGenerateEdit": "DreamLite Generate/Edit",
    "DreamLiteClearCache": "DreamLite Clear Cache",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
