import cv2
import os
import tempfile
import streamlit as st

def handle_video(uploaded_file, tracker):
    st.subheader("📹 Detecting items from video")
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("❌ Failed to open video file.")
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 24  # fallback FPS
        update_interval_frames = int(fps * 3)
        progress_bar = st.progress(0)
        video_placeholder = st.empty()
        summary_placeholder = st.empty()

        def frame_generator():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()

        last_update_frame = 0

        for idx, (annotated_frame, _) in enumerate(
            tracker.track_video_stream(frame_generator(), tracker.confidence_threshold)
        ):
            progress = min((idx + 1) / total_frames, 1.0)
            progress_bar.progress(progress)
            video_placeholder.image(annotated_frame, channels="BGR", width=320)

            if idx == 0 or idx - last_update_frame >= update_interval_frames:
                output_stats = tracker.get_output_stats()
                if not output_stats.empty:
                    summary_placeholder.subheader("📦 Item summary (updating...)")
                    summary_placeholder.dataframe(output_stats, use_container_width=True)
                else:
                    summary_placeholder.info("🔍 Processing... waiting for detections.")
                last_update_frame = idx

        progress_bar.empty()
        os.remove(tfile.name)

    except Exception as e:
        st.error(f"❌ Failed to process video: {e}")
        st.stop()