import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class VisionPlatformSettings(BaseSettings):
    # ─── 1. CORE ENTERPRISE SYSTEM INFRASTRUCTURE ───
    PROJECT_NAME: str = "VisionForge AI Platform"
    VERSION: str = "9.0.0"
    API_V1_STR: str = "/api/v9"
    ENVIRONMENT: str = Field(default="production", validation_alias="ENVIRONMENT")
    
    # ─── 2. INDUSTRIAL EDGE DETECTORS & HARDWARE ACCELERATION ───
    # TensorRT / OpenVINO / CUDA Context Engines
    ALLOW_HARDWARE_FALLBACK: bool = True
    MINIMUM_INFERENCE_FPS: float = 30.0
    BATCH_INFERENCE_SIZE: int = 4
    INT8_QUANTIZATION_ENABLED: bool = True
    
    # ─── 3. STATISTICAL PROCESS CONTROL (SPC) & ANOMALY THRESHOLDS ───
    # العتبة الحرجة لحساب مسافات المتجهات في PatchCore / PaDiM
    PATCHCORE_DISTANCE_THRESHOLD: float = 12.5
    # معامل الـ Lambda لحساب مخطط الـ EWMA الإحصائي للتنبؤ بانحراف جودة الخط
    SPC_EWMA_LAMBDA: float = 0.20
    # نسبة اليقين الدنيا لقبول معايرة النموذج (Confidence Temperature Threshold)
    CALIBRATION_TEMPERATURE: float = 1.4
    ACTIVE_LEARNING_CONFIDENCE_TRIGGER: float = 75.0  # إذا قل اليقين عن 75% تذهب الصورة للمراجعة البشرية
    
    # ─── 4. SECURITY, RBAC & PRIVACY TOKENS ───
    SECRET_KEY: str = Field(default="09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 Days token lifespan
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # ─── 5. DISTRIBUTED DATA & BROKERS (DOCKER STACK LAYER) ───
    POSTGRES_SERVER: str = Field(default="localhost")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="visionforge_secure_pass")
    POSTGRES_DB: str = Field(default="visionforge_prod")
    
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    RABBITMQ_URL: str = Field(default="amqp://guest:guest@localhost:5672//")
    
    # ─── 6. AUTOMATED MLOPS AUTOMATION ───
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")
    MODEL_REGISTRY_PATH: str = Field(default="/opt/visionforge/models")

    # التحقق الديناميكي من صحة البيئة التشغيلية
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_env_scope(cls, v: str) -> str:
        allowed = ["development", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"Operational mode must be within {allowed}")
        return v.lower()

    # دعم قراءة الإعدادات من ملف خارجي .env تلقائياً إذا وُجد
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

# Instantiate single thread-safe operational settings object
settings = VisionPlatformSettings()