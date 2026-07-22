import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any

# استيراد الخدمات الصناعية والذكية
from services.inspection_engine import IndustrialInspectionEngine
from services.inference_fusion import HybridInferenceFusionEngine
from services.config import settings  # استدعاء ملف الإعدادات المركزي المطور

app = FastAPI(
    title=f"⚙️ {settings.PROJECT_NAME} Core Kernel", 
    version=settings.VERSION
)

# Dynamic Dependency Injections من خلال الإعدادات المركزية
inspector = IndustrialInspectionEngine()
# تمرير الـ Temperature المضبط إحصائياً من ملف الـ Config مباشرة
fusion_engine = HybridInferenceFusionEngine()

class VisionForgeInspectionPayload(BaseModel):
    product_passport_id: str
    dimensions: Dict[str, float]
    ocr_data: Dict[str, str]
    surface_analytics: Dict[str, Any]
    missing_components: Dict[str, Any]
    intelligence_layer: Dict[str, Any]
    execution_latency_ms: float

@app.post("/api/v9/inspect-package", response_model=VisionForgeInspectionPayload, status_code=status.HTTP_200_OK)
async def process_industrial_inspection_pipeline(file: UploadFile = File(...)):
    start_time = time.time()
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid manufacturing stream asset. Visual frame stream required."
        )
        
    try:
        # 1. قراءة الفريم وتحويله إلى OpenCV Mat
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Corrupted image artifact bytes.")
        
        # ─── المرحلة الثانية: AI Image Enhancement Pipeline ───
        enhanced_frame = inspector.enhance_frame(frame)
        
        # ─── الطبقة الثانية: Industrial Inspection Execution ───
        dimensions = inspector.execute_dimensional_inspection(enhanced_frame)
        ocr_data = inspector.execute_ocr_inspection(enhanced_frame)
        component_data = inspector.detect_missing_components(enhanced_frame, target_screws=4)
        surface_data = inspector.execute_color_and_texture(enhanced_frame)
        
        # التحقق المعياري للأبعاد بناءً على الـ Constraints الميكانيكية
        dim_status = "PASS" if 28.0 <= dimensions["width_mm"] <= 32.0 else "OUT_OF_BOUNDS"
        
        # ─── الطبقة الثالثة: AI Intelligence Core (Inference Fusion & Unsupervised PatchCore) ───
        # دمج الـ YOLOv11 والـ MobileNetV3 مع الـ Root Cause Analysis (RCA) في خطوة واحدة
        decision_core, localizations = await fusion_engine.execute_fusion_pipeline(
            enhanced_frame, dim_status, component_data["status"]
        )
        
        # محاكاة حساب مسافات التنسورات لـ PatchCore للـ Unknown Defects
        mock_embeddings = np.random.randn(10)
        # حساب المسافة الإقليدية ومقارنتها بالـ Threshold الديناميكي من الـ Config
        patchcore_score = float(np.linalg.norm(mock_embeddings))
        is_unknown_defect = patchcore_score > settings.PATCHCORE_DISTANCE_THRESHOLD
        
        # تقييم الـ Active Learning Queue بناءً على عتبة اليقين في الإعدادات
        needs_human_review = decision_core["calibrated_probability"] < settings.ACTIVE_LEARNING_CONFIDENCE_TRIGGER
        
        latency_ms = (time.time() - start_time) * 1000
        
        return VisionForgeInspectionPayload(
            product_passport_id=f"PASSPORT_ID_{uuid_generator()}",
            dimensions=dimensions,
            ocr_data=ocr_data,
            surface_analytics=surface_data,
            missing_components=component_data,
            intelligence_layer={
                "root_cause_analysis": decision_core["root_cause"],
                "patchcore_anomaly_score": round(patchcore_score, 2),
                "is_unknown_defect_pattern": is_unknown_defect,
                "active_learning_routing": {
                    "data_drift_detected": patchcore_score > 14.0,  # كمثال إحصائي لاكتشاف الـ Drift
                    "active_learning_queue_flag": needs_human_review,
                    "action_required": "ROUTE_TO_ENGINEER_REVIEW" if needs_human_review else "SYSTEM_AUTO_LOG"
                },
                "calibrated_metrics": {
                    "raw_confidence_pct": decision_core["raw_confidence"],
                    "calibrated_probability_pct": decision_core["calibrated_probability"],
                    "all_classes_distribution": decision_core["all_classes_distribution"]
                }
            },
            execution_latency_ms=round(latency_ms, 2)
        )
        
    except Exception as ex:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(ex).__name__}: {str(ex)}"
        )

def uuid_generator() -> str:
    import uuid
    return str(uuid.uuid4()).upper()[:8]