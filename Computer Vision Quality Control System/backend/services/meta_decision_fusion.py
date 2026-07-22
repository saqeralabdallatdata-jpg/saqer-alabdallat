import numpy as np
from typing import List, Dict, Tuple

class MetaDecisionFusionEngine:
    @staticmethod
    def calculate_entropy(probabilities: List[float]) -> float:
        """Calculates mathematical Shannon Entropy to estimate prediction uncertainty"""
        probs = np.array(probabilities)
        probs = probs / np.sum(probs)  # Normalization
        return float(-np.sum(probs * np.log2(probs + 1e-9)))

    def fuse_and_evaluate(self, camera_outputs: List[Dict], anomaly_distance: float) -> Tuple[str, float, float, str, bool]:
        """Executes a strict meta-decision vote over spatial multi-camera inputs and PatchCore metrics"""
        # حساب متوسط اليقين للكاميرات
        all_confidences = [c["raw_confidence"] for c in camera_outputs]
        system_uncertainty = self.calculate_entropy(all_confidences)
        
        # الـ Threshold المعتمد لخوارزمية الـ PatchCore
        is_unknown_defect = anomaly_distance > 11.5 
        
        votes_reject = 0
        detected_faults = []
        
        for c in camera_outputs:
            if "Perfect Quality" not in c["detected_anomalies"]:
                votes_reject += 1
                detected_faults.extend(c["detected_anomalies"])
                
        # القرار النهائي بالتصويت الهجين المعزز بكاشف العيوب المجهولة
        if votes_reject >= 2 or is_unknown_defect:
            verdict = "🔴 REJECTED (Defective)"
            primary_cause = detected_faults[0] if detected_faults else "Unknown Anomaly Pattern (PatchCore Alert)"
        else:
            verdict = "🟢 PASSED (Optimal)"
            primary_cause = "None"
            
        return verdict, system_uncertainty, anomaly_distance, primary_cause, is_unknown_defect