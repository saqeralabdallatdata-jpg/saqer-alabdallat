import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# إعدادات الصفحة العامة للمنصة الهندسية المحدثة v9.0
st.set_page_config(page_title="VisionForge AI Platform v9.0", layout="wide", page_icon="⚙️")

# ربط الـ URL بالإصدار v9 الجديد المتوافق مع الـ Backend والـ Config
BACKEND_URL = "http://127.0.0.1:8000/api/v9/inspect-package"

st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>⚙️ VisionForge AI Industrial Control Center v9.0</h1>", unsafe_allow_html=True)

# محاكاة قاعدة بيانات العيوب للـ Pareto Analysis (الطبقة الرابعة)
mock_defect_db = {"Missing Screw": 45, "Surface Scratch": 22, "Texture Anomaly": 12, "Incorrect Label OCR": 5, "Diameter Expansion": 3}

col_upload, col_telemetry = st.columns([1, 1])

with col_upload:
    st.subheader("📸 Frame Injection Layer")
    uploaded_image = st.file_uploader("📥 Inject Real-Time Production Asset Frame:", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
        with st.spinner("Decoding multidimensional tensors & applying CLAHE..."):
            try:
                # إرسال الطلب لـ API الـ Backend المطور
                res = requests.post(BACKEND_URL, files=files)
                
                if res.status_code == 200:
                    data = res.json()
                else:
                    st.error(f"🚨 Core Exception (Status {res.status_code}): {res.text}")
                    st.stop()
            except Exception as e:
                st.error("🚨 Gateway Timeout: Core AI Inspection Microservice is Offline.")
                st.stop()
                
        if res.status_code == 200:
            intel = data["intelligence_layer"]
            st.image(uploaded_image, caption=f"Active Digital Product Passport: {data['product_passport_id']}", use_container_width=True)
            
            # عرض موجه الـ Active Learning Queue المتوافق مع الـ Backend الجديد
            if intel["active_learning_routing"]["active_learning_queue_flag"]:
                st.warning(f"⚠️ Sent to Active Learning Queue for Human Verification! Reason: Calibrated Probability is below threshold.")

with col_telemetry:
    if uploaded_image and 'data' in locals():
        st.subheader("🔍 Production Inspection Matrix")
        
        # استخراج حالة الـ Missing Components ديناميكياً
        screws_counted = data['missing_components'].get('screws_counted', 0)
        comp_status = data['missing_components'].get('status', 'UNKNOWN')
        
        # استخراج بيانات السطح (اللون والنسيج) المتوافقة مع مخرجات الـ API الجديد
        surface_status = data['surface_analytics'].get('status', 'NOMINAL')
        
        st.markdown(f"""
        - **Extracted Manufacturing Texts (OCR):** `{data['ocr_data']['extracted_text']}`
        - **Component Verification:** `{screws_counted} Screws Detected` ({comp_status})
        - **Calculated Object Diameter:** `{data['dimensions']['width_mm']} mm`
        - **Surface Texture Analytics Status:** `{surface_status}`
        """)
        
        st.markdown("---")
        st.subheader("🧠 Intelligence Layer Diagnostics (RCA)")
        
        rca = intel["root_cause_analysis"]
        calibrated_metrics = intel.get("calibrated_metrics", {})
        
        # تعديل قراءة الـ confidence لتطابق الهيكل الجديد (calibrated_probability_pct)
        confidence_pct = calibrated_metrics.get("calibrated_probability_pct", 100.0)
        
        st.error(f"### Likely Fault Root Cause: `{rca['likely_cause']}` ({confidence_pct}% calibrated certainty)")
        
        # عرض مسافة الـ PatchCore المحدثة للعيوب غير المعروفة
        st.info(f"PatchCore Unsupervised Anomaly Distance Score: `{intel['patchcore_anomaly_score']}`")
        
        if intel["is_unknown_defect_pattern"]:
            st.markdown("⚠️ *Status: Unknown Defect Pattern Detected via Vector Embedding Distance!*")

st.markdown("---")

# ─── الطبقة الرابعة: INDUSTRIAL ANALYTICS DISPLAY (PARETO CHART) ───
st.subheader("📊 Plant-Floor Pareto Chart Analytics (Defect Distribution)")
pareto_data = [
    {"Defect": "Missing Screw", "Count": 45, "Cumulative %": 51.7},
    {"Defect": "Surface Scratch", "Count": 22, "Cumulative %": 77.0},
    {"Defect": "Texture Anomaly", "Count": 12, "Cumulative %": 90.8},
    {"Defect": "Incorrect Label OCR", "Count": 5, "Cumulative %": 96.5},
]
df_p = pd.DataFrame(pareto_data)
fig_pareto = px.bar(df_p, x="Defect", y="Count", title="80/20 Rule Impact Chart", text="Count")
st.plotly_chart(fig_pareto, use_container_width=True)