import asyncio
import numpy as np
from typing import Dict, Tuple, List, Any
from services.confidence_calibrator import ConfidenceCalibrator

class HybridInferenceFusionEngine:
    def __init__(self):
        self.defect_classes = ["Scratch", "Crack", "Missing Part", "Wrong Label", "Missing Screw", "Bent Component"]
        self.calibrator = ConfidenceCalibrator(temperature=1.4)
        self.lock = asyncio.Lock()

    def analyze_root_cause(self, dim_status: str, component_status: str) -> Dict[str, Any]:
        """Root Cause Analysis (RCA) Engine correlating hardware defects to factory components"""
        if component_status == "MISSING_COMPONENT_ALERT":
            return {"likely_cause": "Pneumatic Feeder Injector Node 4", "rca_confidence": 91.5}
        if dim_status == "OUT_OF_BOUNDS":
            return {"likely_cause": "Conveyor Belt Speed Fluctuations (Motor Stall)", "rca_confidence": 78.9}
        return {"likely_cause": "Stationary System Nominal Parameters", "rca_confidence": 100.0}

    async def execute_fusion_pipeline(self, enhanced_frame: np.ndarray, dim_status: str, comp_status: str) -> Tuple[Dict[str, Any], List[dict]]:
        """Simulates simultaneous Multi-Model Forward Pass (YOLOv11 + MobileNetV3) with Decision Fusion"""
        async with self.lock:
            await asyncio.sleep(0.015)  # 15ms simulated deep inference latency
            
            # محاكاة مخرجات الموديل الخام العنيفة
            raw_classification_probs = np.array([0.02, 0.91, 0.04, 0.01, 0.01, 0.01])
            calibrated_probs = self.calibrator.calibrate_logits(raw_classification_probs)
            max_idx = int(np.argmax(calibrated_probs))
            
            primary_defect = self.defect_classes[max_idx]
            
            # استخراج تحليلات الـ Root Cause لايف
            rca_analytics = self.analyze_root_cause(dim_status, comp_status)
            
            # الـ Active Learning Queue Trigger
            calibrated_conf = float(calibrated_probs[max_idx]) * 100
            needs_review = calibrated_conf < 75.0
            
            decision_core = {
                "verdict": "🔴 REJECTED (Defective)" if comp_status != "COMPLETE" or dim_status != "PASS" else "🟢 PASSED (Optimal)",
                "primary_defect": primary_defect if comp_status != "COMPLETE" else "None",
                "raw_confidence": round(float(raw_classification_probs[max_idx]) * 100, 2),
                "calibrated_probability": round(calibrated_conf, 2),
                "all_classes_distribution": {self.defect_classes[i]: round(float(calibrated_probs[i]) * 100, 2) for i in range(len(self.defect_classes))},
                "root_cause": rca_analytics,
                "active_learning_review": needs_review
            }
            
            mock_boxes = [
                {
                    "bbox": [50.0, 120.0, 150.0, 280.0],
                    "defect_type": primary_defect,
                    "segmentation_mask_available": True
                }
            ]
            
            return decision_core, mock_boxes