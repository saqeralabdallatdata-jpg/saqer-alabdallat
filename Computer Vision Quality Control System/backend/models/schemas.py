from pydantic import BaseModel
from typing import List, Dict, Optional

class CameraInspectionFrame(BaseModel):
    camera_id: str
    raw_confidence: float
    detected_anomalies: List[str]

class MetaDecisionAnalysis(BaseModel):
    final_verdict: str  # PASSED / REJECTED
    aggregated_entropy: float
    system_uncertainty_score: float
    primary_fault_cause: str
    is_unknown_anomaly: bool = False  # تفعيل كشف العيوب غير المعروفة سابقاً

class EdgeDeploymentTelemetry(BaseModel):
    active_runtime_engine: str  # TensorRT / ONNX Runtime / CPU OpenVINO
    device_target: str  # NVIDIA Jetson Orin / Intel NUC
    inference_latency_ms: float
    fps: float

class AutonomousQCResponse(BaseModel):
    product_uuid: str
    camera_matrix_results: List[CameraInspectionFrame]
    decision_core: MetaDecisionAnalysis
    hardware_telemetry: EdgeDeploymentTelemetry
    predictive_yield_alert: Optional[str] = None
    timestamp: float