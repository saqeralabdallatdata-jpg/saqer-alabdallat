import cv2
import numpy as np
import easyocr
from typing import Dict, Any, List

class IndustrialInspectionEngine:
    def __init__(self):
        # تفعيل قارئ النصوص والأكواد للصناعة (English + Numbers)
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.pixel_to_mm_ratio = 0.1  # كل 1 بكسل يساوي 0.1 ملم ميكانيكياً

    @staticmethod
    def enhance_frame(img: np.ndarray) -> np.ndarray:
        """Applies Advanced Industrial Image Enhancement Layers sequentially"""
        # 1. Denoising (Gaussian Blur للـ High-frequency noise)
        denoised = cv2.GaussianBlur(img, (3, 3), 0)

        # 2. CLAHE في الـ LAB Color Space لتحسين التباين دون تغيير الهوية البصرية
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 3. Gamma Correction لإصلاح الإضاءة السيئة في بيئة المصنع
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(enhanced_bgr, table)

        # 4. Sharpening Matrix لإبراز حواف التشققات والكسور بدقة ميكروسكوبية
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gamma_corrected, -1, kernel)

        return sharpened

    def execute_dimensional_inspection(self, frame: np.ndarray) -> Dict[str, float]:
        """Calculates precise dimensions using geometric contours: Width, Height, Circularity"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"width_mm": 0.0, "height_mm": 0.0, "circularity": 0.0, "area_mm2": 0.0}
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        
        return {
            "width_mm": round(w * self.pixel_to_mm_ratio, 2),
            "height_mm": round(h * self.pixel_to_mm_ratio, 2),
            "circularity": round(circularity, 2),
            "area_mm2": round(area * (self.pixel_to_mm_ratio ** 2), 2)
        }

    def execute_ocr_inspection(self, frame: np.ndarray) -> Dict[str, str]:
        """Extracts Expiry Dates, Serial Numbers, and Lot Codes from the assembly line"""
        results = self.reader.readtext(frame)
        text_output = " ".join([res[1] for res in results])
        return {
            "extracted_text": text_output,
            "expiry_status": "VALID" if "EXP" in text_output or len(text_output) > 2 else "MISSING_OR_INVALID"
        }

    def detect_missing_components(self, frame: np.ndarray, target_screws: int = 4) -> Dict[str, Any]:
        """Checks for missing screws/nuts via Hough Circles"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                  param1=50, param2=30, minRadius=5, maxRadius=25)
        detected_screws = len(circles[0]) if circles is not None else 0
        return {
            "screws_counted": detected_screws,
            "status": "COMPLETE" if detected_screws >= target_screws else "MISSING_COMPONENT_ALERT"
        }
    def execute_color_and_texture(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Industrial Surface Analytics:
        - Color distribution analysis
        - Brightness measurement
        - Texture roughness estimation
        """

        # استخراج متوسط اللون بصيغة BGR
        mean_bgr = cv2.mean(frame)[:3]

        # تحويل إلى grayscale لتحليل النسيج
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # قياس خشونة السطح باستخدام تباين Laplacian
        texture_variance = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )

        # قياس الإضاءة
        brightness = float(np.mean(gray))

        if texture_variance > 200:
            surface_condition = "HIGH_TEXTURE_VARIATION"
        elif texture_variance > 50:
            surface_condition = "MEDIUM_TEXTURE_VARIATION"
        else:
            surface_condition = "SMOOTH_SURFACE"

        return {
            "dominant_color": {
                "blue": round(mean_bgr[0], 2),
                "green": round(mean_bgr[1], 2),
                "red": round(mean_bgr[2], 2)
            },
            "brightness_level": round(brightness, 2),
            "texture_score": round(texture_variance, 2),
            "surface_condition": surface_condition
        }