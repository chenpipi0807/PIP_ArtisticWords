import os
import sys

# Add the current directory to sys.path to ensure modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import node classes
from .nodes.artistic_text_node import ArtisticTextNode
from .nodes.preview_node import TextPreviewNode
from .nodes.svg_recorder_node import PIPSVGRecorder
from .nodes.PIP_artistic_words_fusion import PIPArtisticWordsFusion
from .nodes.PIP_ColorPicker import PIPColorPicker
from .nodes.PIP_AdvancedColorAnalyzer import PIPAdvancedColorAnalyzer, PIPColorWheel

def _normalize_image_output(value):
    import torch
    if not isinstance(value, torch.Tensor):
        return value
    tensor = value
    if tensor.dim() == 3 and tensor.shape[-1] in (3, 4):
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[-1] == 4:
        tensor = tensor[..., :3]
    return tensor.contiguous() if tensor.dim() == 4 else tensor

def _wrap_node_image_outputs(node_class):
    import functools
    function_name = getattr(node_class, "FUNCTION", None)
    if not function_name or not hasattr(node_class, function_name):
        return node_class
    return_types = getattr(node_class, "RETURN_TYPES", ())
    return_types = return_types if isinstance(return_types, (list, tuple)) else []
    original = getattr(node_class, function_name)
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if isinstance(result, tuple):
            return tuple(
                _normalize_image_output(item) if i < len(return_types) and return_types[i] == "IMAGE" else item
                for i, item in enumerate(result)
            )
        return _normalize_image_output(result)
    setattr(node_class, function_name, wrapped)
    return node_class

for _node_class in (ArtisticTextNode, TextPreviewNode, PIPSVGRecorder, PIPArtisticWordsFusion, PIPColorPicker, PIPAdvancedColorAnalyzer, PIPColorWheel):
    _wrap_node_image_outputs(_node_class)

# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PIP Artistic Text Generator": ArtisticTextNode,
    "PIP Text Preview": TextPreviewNode,
    "PIP SVG Recorder": PIPSVGRecorder,
    "PIP ArtisticWords Fusion": PIPArtisticWordsFusion,
    "PIP ColorPicker": PIPColorPicker,
    "PIPAdvancedColorAnalyzer": PIPAdvancedColorAnalyzer,
    "PIPColorWheel": PIPColorWheel
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP Artistic Text Generator": "PIP Artistic Text Generator",
    "PIP Text Preview": "PIP Text Preview",
    "PIP SVG Recorder": "PIP SVG Recorder",
    "PIP ArtisticWords Fusion": "PIP ArtisticWords Fusion",
    "PIP ColorPicker": "🔴 PIP 颜色拾取",
    "PIPAdvancedColorAnalyzer": "📊 PIP 高级颜色分析",
    "PIPColorWheel": "🎨 PIP 色轮"
}
