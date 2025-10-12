# =============================================================================
# IMPORTS
# =============================================================================
import cv2          # OpenCV for video capture and frame processing
import os           # Operating system operations (file deletion)
import tempfile     # Create temporary files for uploaded videos
import streamlit as st  # Streamlit for UI components and progress tracking

# =============================================================================
# VIDEO HANDLER FUNCTION
# =============================================================================
def handle_video(uploaded_file, tracker):
    """
    Handles video upload and real-time processing in the Streamlit UI.
    
    This function:
    1. Saves uploaded video to a temporary file
    2. Opens video with OpenCV VideoCapture
    3. Processes frames one-by-one using a generator
    4. Displays real-time progress with annotated frames
    5. Updates statistics periodically during processing
    6. Cleans up temporary file after completion
    
    Args:
        uploaded_file: Streamlit UploadedFile object (video file from user)
        tracker: InventoryTracker instance (can be detection or segmentation model)
    
    Note:
        - Works with both detection and segmentation models
        - Shows live preview during processing
        - Updates statistics every 3 seconds (configurable)
        - Uses ByteTrack for consistent object tracking across frames
    """
    # Display section header
    st.subheader("📹 Detecting items from video")
    
    try:
        # =====================================================================
        # STEP 1: SAVE UPLOADED VIDEO TO TEMPORARY FILE
        # =====================================================================
        # OpenCV VideoCapture requires a file path, not a file-like object
        # Create a temporary file that won't be deleted immediately (delete=False)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        
        # Write the uploaded video bytes to the temporary file
        tfile.write(uploaded_file.read())
        
        # Close the file to ensure all data is written
        # File remains on disk due to delete=False
        tfile.close()
        
        # =====================================================================
        # STEP 2: OPEN VIDEO WITH OPENCV
        # =====================================================================
        # Create VideoCapture object to read frames from the saved file
        cap = cv2.VideoCapture(tfile.name)

        # Verify the video file opened successfully
        if not cap.isOpened():
            st.error("❌ Failed to open video file.")
            st.stop()  # Stop execution if video can't be opened

        # =====================================================================
        # STEP 3: GET VIDEO PROPERTIES
        # =====================================================================
        # Get total number of frames in the video for progress calculation
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get frames per second (FPS) for update interval calculation
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 24  # Fallback FPS if video metadata is missing

        # =====================================================================
        # STEP 4: SETUP UI COMPONENTS FOR LIVE UPDATES
        # =====================================================================
        # Calculate how many frames equal 3 seconds for periodic updates
        # Example: 30fps × 3 seconds = update every 90 frames
        update_interval_frames = int(fps * 3)
        
        # Create progress bar (0-100%) for tracking video processing
        progress_bar = st.progress(0)
        
        # Create placeholders for live-updating content
        video_placeholder = st.empty()    # For showing annotated frames
        summary_placeholder = st.empty()  # For showing statistics table

        # =====================================================================
        # STEP 5: DEFINE FRAME GENERATOR FUNCTION
        # =====================================================================
        def frame_generator():
            """
            Generator function that yields video frames one at a time.
            
            This approach is memory-efficient as it doesn't load the entire
            video into memory. Frames are yielded one-by-one for processing.
            
            Yields:
                numpy.ndarray: Video frame in BGR format (OpenCV standard)
            """
            # Continue reading while video is open
            while cap.isOpened():
                # Read the next frame
                # ret: Boolean indicating success
                # frame: NumPy array containing the frame data
                ret, frame = cap.read()
                
                # Break loop if we've reached the end of the video
                if not ret:
                    break
                
                # Yield the frame for processing
                yield frame
            
            # Release video capture when done
            # Frees system resources and closes the video file
            cap.release()

        # =====================================================================
        # STEP 6: PROCESS VIDEO FRAMES WITH TRACKING
        # =====================================================================
        # Track when we last updated the statistics display
        last_update_frame = 0

        # Process video using tracker's stream processing method
        # track_video_stream() yields (annotated_frame, live_summary) for each frame
        # enumerate() gives us the frame index for progress tracking
        for idx, (annotated_frame, _) in enumerate(
            tracker.track_video_stream(
                frame_generator(),           # Generator yielding frames
                tracker.confidence_threshold  # YOLO confidence threshold
            )
        ):
            # =================================================================
            # STEP 6.1: UPDATE PROGRESS BAR
            # =================================================================
            # Calculate progress as a percentage (0.0 to 1.0)
            # min() ensures we don't exceed 100% due to frame count inaccuracies
            progress = min((idx + 1) / total_frames, 1.0)
            progress_bar.progress(progress)
            
            # =================================================================
            # STEP 6.2: DISPLAY CURRENT ANNOTATED FRAME
            # =================================================================
            # Update the video placeholder with the latest processed frame
            # width=320 keeps the preview at a reasonable size
            # channels="BGR" tells Streamlit to handle OpenCV's BGR format
            video_placeholder.image(
                annotated_frame, 
                channels="BGR", 
                width=320
            )

            # =================================================================
            # STEP 6.3: PERIODICALLY UPDATE STATISTICS TABLE
            # =================================================================
            # Update statistics either on:
            #   1. First frame (idx == 0)
            #   2. After update_interval_frames have passed since last update
            # This prevents excessive UI updates that could slow down processing
            if idx == 0 or idx - last_update_frame >= update_interval_frames:
                # Get current aggregated statistics from tracker
                output_stats = tracker.get_output_stats()
                
                # Display statistics if we have detections
                if not output_stats.empty:
                    # Show updating header to indicate live processing
                    summary_placeholder.subheader("📦 Item summary (updating...)")
                    # Display the statistics DataFrame
                    summary_placeholder.dataframe(
                        output_stats, 
                        use_container_width=True
                    )
                else:
                    # Show info message if no detections yet
                    summary_placeholder.info("🔍 Processing... waiting for detections.")
                
                # Update the last update frame counter
                last_update_frame = idx

        # =====================================================================
        # STEP 7: CLEANUP AFTER PROCESSING
        # =====================================================================
        # Remove the progress bar once processing is complete
        progress_bar.empty()
        
        # Delete the temporary video file to free disk space
        # The file path is stored in tfile.name
        os.remove(tfile.name)

    except Exception as e:
        # =====================================================================
        # ERROR HANDLING
        # =====================================================================
        # Catch any errors during video processing and display to user
        # Common errors:
        #   - Corrupt video file
        #   - Unsupported video codec
        #   - Model inference failure
        #   - Memory issues with large videos
        #   - Disk space issues (temporary file)
        st.error(f"❌ Failed to process video: {e}")
        st.stop()  # Stop execution to prevent further errors