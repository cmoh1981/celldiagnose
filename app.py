"""
CellDiagnose-AI: Main Streamlit Application
============================================
Professional web-based diagnostic tool for brightfield cell microscopy images.
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Local imports
import auth
from utils import (
    ImagePreprocessor,
    ConfluencyAnalyzer,
    AnomalyDetector,
    create_diagnostic_summary
)
from models import ModelLoader, CellClassifier
from config import get_config, get_cell_type_info, CELL_TYPE_INFO

# Try to import enhanced anomaly detection
try:
    from anomaly_detection import CellAnomalyDetector
    ENHANCED_ANOMALY_AVAILABLE = True
except ImportError:
    ENHANCED_ANOMALY_AVAILABLE = False


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="CellDiagnose-AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling - Anti-shake fix
st.markdown("""
<style>
    /* CRITICAL: Force stable viewport width */
    html {
        overflow-y: scroll !important;
        scrollbar-gutter: stable !important;
    }

    body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        overflow-x: hidden !important;
        width: 100% !important;
        max-width: 100vw !important;
    }

    .main .block-container {
        padding: 2rem;
        max-width: 100%;
        width: 100%;
        box-sizing: border-box;
    }

    /* Force scrollbar always visible */
    ::-webkit-scrollbar {
        width: 12px !important;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 6px;
    }

    /* CRITICAL: Disable ALL animations and transitions */
    *, *::before, *::after {
        transition: none !important;
        animation: none !important;
        transform: none !important;
        scroll-behavior: auto !important;
    }

    /* Stabilize columns with fixed flex */
    [data-testid="column"] {
        min-width: 0 !important;
        flex-shrink: 0 !important;
        contain: layout style !important;
    }

    /* Stabilize horizontal blocks */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        gap: 1rem !important;
        contain: layout !important;
    }

    /* Fixed height for metrics to prevent reflow */
    [data-testid="stMetric"] {
        min-height: 90px !important;
        contain: layout style !important;
    }

    [data-testid="stMetricValue"] {
        min-height: 36px !important;
    }

    /* Stabilize expanders */
    [data-testid="stExpander"] {
        contain: layout style !important;
    }

    /* Prevent image reflow */
    [data-testid="stImage"] {
        contain: layout !important;
    }

    [data-testid="stImage"] img {
        max-width: 100% !important;
        height: auto !important;
    }

    /* Progress bar stability */
    .stProgress > div > div {
        transition: none !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader()
        CellClassifier.load_cell_types_from_config()

    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    if 'results' not in st.session_state:
        st.session_state.results = None

    if 'history' not in st.session_state:
        st.session_state.history = []


# =============================================================================
# Components
# =============================================================================

def render_header():
    """Render application header."""
    st.title("🔬 CellDiagnose-AI")
    st.markdown("_Automated diagnostic analysis for brightfield cell microscopy_")
    st.markdown("---")


def render_sidebar() -> Dict[str, Any]:
    """Render sidebar and return configuration."""
    with st.sidebar:
        st.markdown("## 🎛️ Configuration")
        st.markdown("---")

        # Image upload
        st.markdown("### 📤 Image Upload")
        uploaded_file = st.file_uploader(
            "Upload microscopy image",
            type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
            help="Supported: JPG, PNG, TIF (brightfield microscopy)"
        )

        # Run button (Moved up)
        run_analysis = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
            key="run_analysis_top"
        )

        st.markdown("---")

        # Analysis settings
        st.markdown("### ⚙️ Analysis Settings")

        segmentation_method = st.selectbox(
            "Segmentation Method",
            options=["OpenCV (Fast)", "U-Net (Precise)"],
            index=0,
            help="OpenCV for quick analysis, U-Net for precise segmentation"
        )

        classification_model = st.selectbox(
            "Classification Model",
            options=["EfficientNet-B0", "ResNet-50"],
            index=0
        )

        st.markdown("---")

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            contamination_sensitivity = st.slider(
                "Contamination Sensitivity",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Lower = more sensitive"
            )

            show_confidence = st.checkbox("Show confidence scores", value=True)
            show_debug = st.checkbox("Show debug info", value=False)

        st.markdown("---")

        # Info
        st.markdown("---")
        st.markdown("### ℹ️ System Info")
        device = st.session_state.model_loader.device_info
        st.caption(f"Device: {device}")

        # Supported cell types
        with st.expander("📋 Supported Cell Types"):
            config = get_config()
            cell_types = config.data.cell_types if config.data.cell_types else list(CELL_TYPE_INFO.keys())[:10]
            for cell_type in cell_types[:10]:
                info = get_cell_type_info(cell_type)
                st.caption(f"**{cell_type}**: {info['description']}")

        # Contact Info
        st.markdown("---")
        st.markdown("""
        <div style="padding: 1rem; background: #eff6ff; border-radius: 12px; border: 1px solid #dbeafe; text-align: center;">
            <p style="margin: 0; font-size: 0.85rem; color: #1e40af; font-weight: 500;">
                ✉️ Support & Inquiries
            </p>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem; color: #3b82f6;">
                <a href="mailto:brownbio.ocm@gmail.com" style="color: #3b82f6; text-decoration: none;">
                    brownbio.ocm@gmail.com
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    return {
        'uploaded_file': uploaded_file,
        'segmentation_method': segmentation_method,
        'classification_model': classification_model,
        'contamination_sensitivity': contamination_sensitivity,
        'show_confidence': show_confidence,
        'show_debug': show_debug,
        'run_analysis': run_analysis
    }


def check_access():
    """Check if user has access (active trial or registered)."""
    is_active, days_left, _ = auth.get_trial_status()
    is_registered = auth.is_user_registered()
    
    if not is_active and not is_registered:
        st.error("🚫 Trial Expired")
        st.markdown(f"""
        ### Your 1-week free trial has expired.
        Please sign up to continue using CellDiagnose-AI and all its features for free!
        
        We use sign-ups only to track the number of active researchers using our tool.
        """)
        
        with st.form("signup_form"):
            st.markdown("### 📝 Sign Up")
            username = st.text_input("Username", placeholder="e.g. Dr. Smith")
            email = st.text_input("Email", placeholder="e.g. smith@lab.org")
            submit = st.form_submit_button("Sign Up Now")
            
            if submit:
                if username and email:
                    auth.register_user(username, email)
                    st.success("✅ Registration successful! Refreshing...")
                    st.rerun()
                else:
                    st.warning("Please provide both username and email.")
        return False
    
    # In sidebar, show status
    with st.sidebar:
        st.markdown("---")
        if is_registered:
            st.success("✅ Full Version Active (Registered)")
        else:
            st.info(f"⏳ Trial Mode: {days_left} days remaining")
            if st.button("Sign Up for Permanent Access"):
                with st.expander("📝 Sign Up Now", expanded=True):
                    with st.form("manual_signup"):
                        u = st.text_input("Username")
                        e = st.text_input("Email")
                        s = st.form_submit_button("Complete Registration")
                        if s and u and e:
                            auth.register_user(u, e)
                            st.success("Registration complete!")
                            st.rerun()
    return True


def run_analysis_pipeline(image: np.ndarray, config: Dict) -> Dict[str, Any]:
    """Execute analysis pipeline."""
    results = {}

    # Progress
    progress = st.progress(0, text="Initializing...")

    # 1. Confluency Analysis
    progress.progress(20, text="Analyzing cell confluency...")

    if "U-Net" in config['segmentation_method']:
        seg_model = st.session_state.model_loader.get_segmentation_model('unet')
        seg_result = seg_model.predict(image)
        results['segmentation'] = {
            'mask': seg_result['mask'],
            'confluency_percent': seg_result['confluency_percent'],
            'cell_count_estimate': 0,
            'method_used': 'U-Net',
            'is_mock': seg_result['is_mock']
        }
        results['segmentation']['overlay'] = create_overlay(image, seg_result['mask'])
    else:
        analyzer = ConfluencyAnalyzer(method="adaptive_threshold")
        seg_result = analyzer.analyze(image)
        results['segmentation'] = seg_result
        results['segmentation']['is_mock'] = False

    progress.progress(50, text="Classifying cell type...")

    # 2. Classification
    backbone = 'efficientnet' if 'EfficientNet' in config['classification_model'] else 'resnet50'
    classifier = st.session_state.model_loader.get_classification_model(backbone)
    results['classification'] = classifier.predict(image)

    progress.progress(75, text="Detecting anomalies...")

    # 3. Anomaly Detection
    detector = AnomalyDetector(contamination_threshold=config['contamination_sensitivity'])
    results['anomaly'] = detector.analyze(image)

    progress.progress(85, text="Running enhanced health check...")

    # 4. Enhanced Anomaly Detection (Autoencoder-based)
    if ENHANCED_ANOMALY_AVAILABLE:
        try:
            if 'enhanced_detector' not in st.session_state:
                st.session_state.enhanced_detector = CellAnomalyDetector()

            enhanced_result = st.session_state.enhanced_detector.analyze(image)
            results['enhanced_anomaly'] = enhanced_result
            results['original_image'] = image
        except Exception as e:
            print(f"Enhanced anomaly detection error: {e}")
            results['enhanced_anomaly'] = None

    progress.progress(95, text="Generating report...")

    # 5. Summary
    results['summary'] = create_diagnostic_summary(
        results['segmentation'],
        results['classification'],
        results['anomaly']
    )

    progress.progress(100, text="Complete!")

    return results


def create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create visualization overlay."""
    overlay = image.copy()
    cell_overlay = np.zeros_like(image)
    cell_overlay[:, :, 1] = mask

    overlay = cv2.addWeighted(overlay, 1, cell_overlay, 0.4, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    return overlay


def render_image_analysis(original: np.ndarray, overlay: np.ndarray, mask: np.ndarray):
    """Render image comparison section."""
    st.markdown("""
    <div style="margin-top: 1rem; margin-bottom: 2rem;">
        <h3 style="color:#0f172a; margin-bottom:1.5rem; display:flex; align-items:center; gap:0.75rem;">
            <span style="font-size:1.8rem;">🖼️</span> Image Analysis Results
        </h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original Image", use_container_width=True)

    with col2:
        st.image(overlay, caption="Segmentation Overlay", use_container_width=True)

    with col3:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        st.image(mask_rgb, caption="AI Generated Mask", use_container_width=True)


def render_diagnostic_report(results: Dict, config: Dict):
    """Render diagnostic report."""
    st.markdown("### 📊 Diagnostic Report")

    summary = results['summary']
    classification = results['classification']

    # Overall Status
    status = summary['overall']['status']
    if status == "Healthy":
        status_class = "success"
        status_icon = "✅"
    else:
        status_class = "warning"
        status_icon = "⚠️"

    st.markdown(f"""
    <div class="report-header">
        <h2 style="margin:0; color:white;">{status_icon} Overall Status: {status}</h2>
        <p style="margin:0.5rem 0 0 0; color:#cbd5e1;">{summary['overall']['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confluency = summary['confluency']['percentage']
        st.metric("Confluency", f"{confluency:.1f}%")
        st.caption(f"Est. {summary['confluency']['cell_count']} cells")

    with col2:
        cell_type = classification['cell_type']
        confidence = classification['cell_type_confidence']
        st.metric("Cell Type", cell_type)
        if config['show_confidence']:
            st.caption(f"{confidence:.1%} Confidence")

    with col3:
        health = classification['health_status']
        health_conf = classification['health_confidence']
        st.metric("Health Status", health)
        if config['show_confidence']:
            st.caption(f"{health_conf:.1%} Confidence")

    with col4:
        contaminated = summary['contamination']['is_contaminated']
        cont_status = "Negative" if not contaminated else "Positive"
        st.metric("Contamination", cont_status)
        st.caption(f"Anomaly Score: {summary['contamination']['score']:.3f}")

    st.markdown("---")

    # Detailed Sections
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📈 Cell Type Probabilities", expanded=False):
            if 'cell_type_probs' in classification:
                for ctype, prob in sorted(
                    classification['cell_type_probs'].items(),
                    key=lambda x: x[1], reverse=True
                ):
                    st.progress(prob, text=f"{ctype}: {prob:.1%}")

    with col2:
        with st.expander("🔍 Contamination Analysis", expanded=False):
            anomaly = results['anomaly']
            st.markdown(f"""
            - **Anomaly Score:** {anomaly['anomaly_score']:.4f}
            - **High-Freq Energy:** {anomaly['high_freq_energy']:.4f}
            - **Texture Irregularity:** {anomaly['texture_irregularity']:.4f}

            **Analysis:** {anomaly['details']}
            """)

    # Enhanced Anomaly Detection (Autoencoder-based)
    if 'enhanced_anomaly' in results:
        st.markdown("---")
        render_enhanced_anomaly_section(results['enhanced_anomaly'], results.get('original_image'))

    # Debug info
    if config['show_debug']:
        with st.expander("🐛 Debug Information"):
            st.json({
                'segmentation_mock': results['segmentation'].get('is_mock'),
                'classification_mock': results['classification'].get('is_mock'),
                'device': st.session_state.model_loader.device_info,
            })


def render_enhanced_anomaly_section(enhanced_result: Dict, original_image: np.ndarray = None):
    """Render enhanced anomaly detection results with heatmap visualization."""
    st.markdown("### 🩺 Cell Health Assessment (AI-Powered)")

    if enhanced_result is None:
        st.info("Enhanced anomaly detection not available. Train the autoencoder model for advanced health monitoring.")
        return

    # Severity color mapping
    severity_colors = {
        'normal': '#10b981',      # Green
        'low': '#fbbf24',         # Yellow
        'medium': '#f97316',      # Orange
        'high': '#ef4444',        # Red
        'critical': '#7c3aed'     # Purple
    }

    severity = enhanced_result['severity']
    color = severity_colors.get(severity, '#64748b')

    # Main status card
    is_anomaly = enhanced_result['is_anomaly']
    status_text = "ABNORMALITY DETECTED" if is_anomaly else "HEALTHY"
    status_icon = "⚠️" if is_anomaly else "✅"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg, {color}22, {color}11);
                border-left:4px solid {color};
                border-radius:8px;
                padding:1.5rem;
                margin-bottom:1rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div>
                <h3 style="margin:0;color:{color};font-size:1.5rem;">{status_icon} {status_text}</h3>
                <p style="margin:0.5rem 0 0 0;color:#cbd5e1;font-weight:400;">
                    Severity: <strong style="color:{color};">{severity.upper()}</strong> |
                    Score: {enhanced_result['anomaly_score']:.4f} |
                    Confidence: {enhanced_result['confidence']:.1%}
                </p>
            </div>
            <div style="text-align:right;">
                <div style="font-size:2.5rem;font-weight:700;color:{color};">
                    {enhanced_result['normalized_score']:.1f}x
                </div>
                <div style="font-size:0.75rem;color:#64748b;">vs baseline</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visualization columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔥 Anomaly Heatmap")
        if 'anomaly_map' in enhanced_result and original_image is not None:
            # Create heatmap overlay
            anomaly_map = enhanced_result['anomaly_map']
            heatmap = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # Resize heatmap to match image
            h, w = original_image.shape[:2]
            heatmap = cv2.resize(heatmap, (w, h))

            # Blend with original
            if len(original_image.shape) == 2:
                original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            else:
                original_rgb = original_image

            overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)
            st.image(overlay, caption="Areas with potential issues highlighted in red/yellow",
                    use_container_width=True)
        else:
            st.info("Heatmap visualization not available")

    with col2:
        st.markdown("#### 📋 Possible Issues")

        if enhanced_result['possible_issues']:
            for issue in enhanced_result['possible_issues']:
                issue_info = {
                    'bacterial': ('🦠', 'Bacterial Contamination', 'Small dark particles detected in background'),
                    'dead_cells': ('💀', 'Dead/Floating Cells', 'Round, bright refractile objects detected'),
                    'stress': ('😰', 'Cell Stress', 'Vacuoles or abnormal morphology observed'),
                    'debris': ('🗑️', 'Debris/Artifacts', 'Non-cellular particles in field of view'),
                    'mycoplasma': ('🔬', 'Possible Mycoplasma', 'Subtle changes - PCR test recommended'),
                    'overgrowth': ('📈', 'Overgrowth', 'High confluency may cause stress')
                }.get(issue, ('❓', issue.title(), 'Unknown condition'))

                st.markdown(f"""
                <div style="background:#f8fafc;border-radius:8px;padding:1rem;margin-bottom:0.5rem;border:1px solid #e2e8f0;">
                    <div style="display:flex;align-items:center;gap:0.5rem;">
                        <span style="font-size:1.5rem;">{issue_info[0]}</span>
                        <div>
                            <strong style="color:#1a1a2e;">{issue_info[1]}</strong>
                            <p style="margin:0;font-size:0.85rem;color:#64748b;">{issue_info[2]}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No specific issues identified")

    # Recommendation box
    st.markdown("#### 💡 Recommendation")
    recommendation = enhanced_result.get('recommendation', 'Continue normal monitoring.')

    rec_bg = '#dcfce7' if severity == 'normal' else '#fef3c7' if severity in ['low', 'medium'] else '#fee2e2'
    rec_border = '#10b981' if severity == 'normal' else '#f59e0b' if severity in ['low', 'medium'] else '#ef4444'

    st.markdown(f"""
    <div style="background:{rec_bg};border:1px solid {rec_border};border-radius:8px;padding:1rem;">
        <p style="margin:0;color:#1a1a2e;font-size:1rem;">{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)


def render_export_section(results: Dict, image: np.ndarray):
    """Render export options."""
    st.markdown("---")
    st.markdown("### 📥 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # JSON Report
        def convert_to_json_serializable(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return None  # Skip arrays
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            return obj
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': convert_to_json_serializable(results['summary']),
            'classification': convert_to_json_serializable({
                k: v for k, v in results['classification'].items()
                if not isinstance(v, np.ndarray)
            }),
            'anomaly': convert_to_json_serializable({
                k: v for k, v in results['anomaly'].items()
                if not isinstance(v, np.ndarray)
            })
        }
        st.download_button(
            "📄 Download JSON Report",
            data=json.dumps(report_data, indent=2),
            file_name=f"celldiagnose_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    with col2:
        # Mask image
        mask = results['segmentation']['mask']
        _, mask_png = cv2.imencode('.png', mask)
        st.download_button(
            "🖼️ Download Mask",
            data=mask_png.tobytes(),
            file_name="segmentation_mask.png",
            mime="image/png",
            use_container_width=True
        )

    with col3:
        # Overlay image
        overlay = results['segmentation']['overlay']
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) if len(overlay.shape) == 3 else overlay
        _, overlay_png = cv2.imencode('.png', overlay_rgb)
        st.download_button(
            "🎨 Download Overlay",
            data=overlay_png.tobytes(),
            file_name="analysis_overlay.png",
            mime="image/png",
            use_container_width=True
        )


def render_welcome():
    """Render welcome screen."""
    st.markdown("""
    <div style="text-align:center; padding:5rem 2rem; background:linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius:30px; margin:2rem 0; box-shadow: inset 0 2px 10px rgba(0,0,0,0.02);">
        <h2 style="color:#0f172a; font-size:2.8rem; margin-bottom:1rem; font-family:'Outfit', sans-serif;">Deep Learning Cell Diagnostics</h2>
        <p style="color:#64748b; font-size:1.3rem; max-width:700px; margin:0 auto; font-weight:300;">
            Upload your brightfield microscopy images to perform automated confluency measurement, 
            cell type classification, and anomaly screening in seconds.
        </p>
        <div style="margin-top: 3rem; display:flex; justify-content:center; gap:2rem;">
            <div style="display:flex; align-items:center; gap:0.5rem; color:#10b981; font-weight:600;">✅ 99.6% Accuracy</div>
            <div style="display:flex; align-items:center; gap:0.5rem; color:#2563eb; font-weight:600;">⚡ Real-time Analysis</div>
            <div style="display:flex; align-items:center; gap:0.5rem; color:#7c3aed; font-weight:600;">🧬 13+ Cell Types</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <h3 style="margin-top:3rem; margin-bottom:1.5rem; color:#1e293b; font-family:'Outfit', sans-serif; text-align:center;">
        <span style="background: var(--primary-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">✨ Powered by Advanced AI</span>
    </h3>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2.5rem; margin-bottom:1rem;">📊</div>
            <h4 style="color:#0f172a; margin-bottom:0.5rem;">Precision Segmentation</h4>
            <p style="color:#64748b; font-size:0.95rem; line-height:1.6;">
                Calculate exact cell coverage using U-Net architectures and adaptive thresholding methods.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🧬</div>
            <h4 style="color:#0f172a; margin-bottom:0.5rem;">Automated Typing</h4>
            <p style="color:#64748b; font-size:0.95rem; line-height:1.6;">
                Instantly identify 13+ cell lines including HeLa, HEK293, and more with high-confidence grading.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🛡️</div>
            <h4 style="color:#0f172a; margin-bottom:0.5rem;">Health Monitoring</h4>
            <p style="color:#64748b; font-size:0.95rem; line-height:1.6;">
                Screen for bacterial contamination and dead cells using texture-aware anomaly detection.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Supported cell types
    st.markdown("### 🧫 Supported Cell Types")

    config = get_config()
    cell_types = config.data.cell_types if config.data.cell_types else list(CELL_TYPE_INFO.keys())[:10]

    cols = st.columns(5)
    for i, cell_type in enumerate(cell_types):
        info = get_cell_type_info(cell_type)
        with cols[i % 5]:
            st.markdown(f"""
            <div class="cell-info" style="margin-bottom:0.5rem;">
                <strong>{cell_type}</strong><br>
                <small style="color:#64748b;">{info['description']}</small>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    # Access check
    if not check_access():
        st.stop()
        
    render_header()

    config = render_sidebar()

    if config['uploaded_file'] is not None:
        image = ImagePreprocessor.load_image(config['uploaded_file'])

        # Run analysis only when button is pressed (one-shot)
        if config['run_analysis']:
            with st.spinner("Running analysis..."):
                results = run_analysis_pipeline(image, config)
                st.session_state.results = results
                st.session_state.analysis_complete = True
                st.session_state.current_filename = config['uploaded_file'].name

                # Add to history (guarded to prevent duplicates)
                if not st.session_state.history or st.session_state.history[-1].get('filename') != config['uploaded_file'].name:
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'filename': config['uploaded_file'].name,
                        'cell_type': results['classification']['cell_type'],
                        'confluency': results['summary']['confluency']['percentage']
                    })

        # Display results if analysis is complete for this file
        if st.session_state.analysis_complete and st.session_state.results:
            results = st.session_state.results

            render_image_analysis(
                image,
                results['segmentation']['overlay'],
                results['segmentation']['mask']
            )

            render_diagnostic_report(results, config)
            render_export_section(results, image)
        else:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.info("👆 Click 'Run Analysis' in the sidebar to begin.")
    else:
        # Clear results if no file uploaded
        if st.session_state.analysis_complete:
            st.session_state.analysis_complete = False
            st.session_state.results = None
        render_welcome()


if __name__ == "__main__":
    main()
