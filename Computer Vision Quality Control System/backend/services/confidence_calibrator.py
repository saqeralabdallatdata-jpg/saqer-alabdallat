import numpy as np

class ConfidenceCalibrator:
    def __init__(self, temperature: float = 1.4):
        self.temperature = temperature

    def calibrate_logits(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """Transforms overconfident softmax outputs via Temperature Scaling"""
        eps = 1e-7
        clipped_probs = np.clip(raw_probabilities, eps, 1.0 - eps)
        
        # 1. Reverse Softmax to recover raw unnormalized logits
        logits = np.log(clipped_probs)
        
        # 2. Scaling down via Temperature Parameter
        scaled_logits = logits / self.temperature
        
        # 3. Re-evaluating smooth, statistically robust probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)
        
        return calibrated_probs