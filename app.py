# Initialize 
import streamlit as st
from py.InventoryTracker import InventoryTracker
from py.handlers.image_handler import handle_image
from py.handlers.video_handler import handle_video

# -------------------------------
# App configuration
# -------------------------------

## APP page setup
st.set_page_config(
    page_title="Automated inventory monitoring",
    page_icon="🛒",
    layout="wide")

### Add logo
def render_logo():
    col1, col2 = st.columns([6, 1])
    with col2:
        st.image("assets/logo.png", width=100)

render_logo()

### Hide original sidebar
hide_sidebar = """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        .css-1y4p8pa {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# -------------------------------
# Initialize tracker in session
# -------------------------------
if 'tracker' not in st.session_state:
    st.session_state.tracker = InventoryTracker()

tracker = st.session_state.tracker

# -------------------------------
# App title
# -------------------------------
# with col_center:
#     st.title("🛒 Inventory detector")
st.markdown(
    """<h1 style='text-align: center; font-size: 48px; width: 100%;'>🛒 Inventory Detector</h1>""",
    unsafe_allow_html=True)
#st.markdown("<br>", unsafe_allow_html=True)
    
# -------------------------------
# APP file uploader
# -------------------------------
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    uploaded_file = st.file_uploader(
        "Upload image or video",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"],
        label_visibility="visible")
    
# -------------------------------
# APP model & inventory level selection
# -------------------------------
with col_center:
    model_label_col, label_mode_col = st.columns([1, 1])  
    with model_label_col:
        st.write("⚙️ Select fine tuned model")
        model_choices = ["models/model_25-08-02.pt", 
                         "models/model_25-08-01.pt", 
                         "models/yolov8n.pt"]
        model_selected = st.selectbox("", options=model_choices, index=0)
        if model_selected != getattr(tracker, "model_path", None):
            tracker.model = tracker.model.__class__(model_selected)
            tracker.model_path = model_selected
            tracker.reset_output_stats()
            st.success(f"✅ Loaded model: {model_selected}")
    with label_mode_col:
        st.write("⚙️ Inventory detection level (use `sku_code` for developer test):")
        tracker.label_mode = st.selectbox(
            "",
            options=["item_name", "category", "sub_category", "brand", "sku_code"],
            index=0)

# -------------------------------
# Helper functions
# -------------------------------
def is_image(file):
    return file.type.startswith("image/")

def is_video(file):
    return file.type.startswith("video/")

# -------------------------------
# App server
# -------------------------------
st.markdown(
    """<h1 style='text-align: center; font-size: 48px; width: 100%;'>🖼️ Detecting items from image</h1>""",
    unsafe_allow_html=True)

if uploaded_file:
    tracker = st.session_state.tracker
    tracker.reset_output_stats()
    if is_image(uploaded_file):
        handle_image(uploaded_file, tracker)
    elif is_video(uploaded_file):
        handle_video(uploaded_file, tracker)
    else:
        st.warning("Unsupported file type.")
        st.stop()
        