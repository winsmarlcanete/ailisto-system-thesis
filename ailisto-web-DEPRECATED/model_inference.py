# model_inference.py
# Minimal placeholder for your model. Replace predict() with actual inference.
import time

class Model:
    def __init__(self, device='cpu', weights_path=None):
        self.device = device
        self.weights_path = weights_path
        # TODO: load your PyTorch model here when ready
        # e.g., self.model = torch.hub.load(...)

    def predict(self, frame):
        """
        Input: frame (BGR numpy array)
        Output: list of detections like:
          [{'label':'student','conf':0.92,'box':[x1,y1,x2,y2]}, ...]
        For now returns empty list (no detections).
        """
        time.sleep(0)  # placeholder
        return []
