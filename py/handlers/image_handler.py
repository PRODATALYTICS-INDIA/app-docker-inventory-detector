import cv2
import numpy as np
from PIL import Image
import streamlit as st

def handle_image(uploaded_file, tracker):
    try:
        # reset stats before processing
        tracker.reset_output_stats()

        # read and convert image
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # run detection
        annotated_frame, _ = tracker.track_picture_stream(frame, tracker.confidence_threshold)

        # show result - annotated image and stats side by side
        col_img, col_table = st.columns([2, 2])  # 2:1 ratio for image/table
        
        with col_img:
            st.image(annotated_frame, caption="", channels="BGR", use_container_width=True)

        with col_table:
            output_stats = tracker.get_output_stats()
            if not output_stats.empty:
                st.dataframe(output_stats, use_container_width=True)
            else:
                st.info("🔍 No items detected.")

    except Exception as e:
        st.error(f"❌ Failed to process image: {e}")
        st.stop()