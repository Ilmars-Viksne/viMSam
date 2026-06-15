from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    name: str = "vit_b"
    device: str = "cuda"

@dataclass
class PromptConfig:
    points: Optional[List[List[int]]] = None
    box: Optional[List[int]] = None

@dataclass
class WorkflowConfig:
    workflow_type: str  # 'single', 'batch', 'video'
    input_uri: str
    output_uri: str
    model: ModelConfig = field(default_factory=ModelConfig)
    prompts: Optional[PromptConfig] = None

    # Options
    show_prompts: bool = False
    save_combined: bool = False  # New Flag: If True, saves Original|Mask|Overlay
    tracking_method: str = "box" # box, centroid, pole
    export_format: str = "csv"   # csv, json, txt

@dataclass
class SegmentationResult:
    success: bool
    count: int = 0
    message: str = ""
