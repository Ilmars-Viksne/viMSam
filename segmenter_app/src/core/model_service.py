import torch
import micro_sam.util
from src.utils import setup_logger

logger = setup_logger(__name__)

class ModelService:
    def __init__(self):
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_name="vit_b"):
        if self.predictor is not None: return
        logger.info(f"Loading MicroSAM ({model_name}) on {self.device}...")
        self.predictor = micro_sam.util.get_sam_model(model_type=model_name, device=self.device)

    def get_predictor(self):
        return self.predictor
