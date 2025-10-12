# =============================================================================
# IMPORTS
# =============================================================================
import cv2           # OpenCV for image processing and color conversion
import numpy as np   # NumPy for array operations
from PIL import Image  # Python Imaging Library for reading uploaded images
import streamlit as st  # Streamlit for UI components

# =============================================================================
# IMAGE HANDLER FUNCTION
# =============================================================================
def handle_image(uploaded_file, tracker):
    """
    Handles single image upload and processing in the Streamlit UI.
    
    This function:
    1. Reads the uploaded image file
    2. Converts it to OpenCV format
    3. Runs YOLO detection/segmentation
    4. Displays annotated results with statistics
    
    Args:
        uploaded_file: Streamlit UploadedFile object (image file from user)
        tracker: InventoryTracker instance (can be detection or segmentation model)
    
    Note:
        - For detection models: Shows boxes + labels + traces
        - For segmentation models: Shows masks + boxes + labels + traces
        - The function automatically adapts based on tracker's model type
    """
    try:
        # =====================================================================
        # STEP 1: RESET STATISTICS
        # =====================================================================
        # Clear any previous tracking data to ensure clean results
        # This resets frame counters, tracked IDs, and confidence scores
        tracker.reset_output_stats()

        # =====================================================================
        # STEP 2: READ AND CONVERT IMAGE
        # =====================================================================
        # Read the uploaded file using PIL (handles various image formats)
        image = Image.open(uploaded_file)
        
        # Convert PIL Image to NumPy array for OpenCV processing
        frame = np.array(image)
        
        # Convert color space from RGB (PIL default) to BGR (OpenCV format)
        # OpenCV uses BGR color ordering, while PIL/most libraries use RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # =====================================================================
        # STEP 3: RUN DETECTION/SEGMENTATION
        # =====================================================================
        # Process the frame with YOLO model
        # Returns:
        #   - annotated_frame: Image with visual overlays (masks/boxes/labels/traces)
        #   - live_summary: Dict of detections (not used here, but available)
        annotated_frame, _ = tracker.track_picture_stream(
            frame, 
            tracker.confidence_threshold
        )

        # =====================================================================
        # STEP 4: DISPLAY RESULTS IN TWO COLUMNS
        # =====================================================================
        # Create a 2-column layout for image and statistics table
        # Both columns have equal width (ratio 2:2)
        col_img, col_table = st.columns([2, 2])
        
        # Left column: Display annotated image
        with col_img:
            # Show the processed image with all annotations
            # channels="BGR" tells Streamlit to handle BGR format correctly
            # use_container_width=True makes image responsive to column width
            st.image(
                annotated_frame, 
                caption="",  # No caption needed
                channels="BGR",  # OpenCV uses BGR format
                use_container_width=True
            )

        # Right column: Display statistics table
        with col_table:
            # Get aggregated detection statistics from tracker
            # Returns pandas DataFrame with columns:
            #   - item_name/category/etc (based on label_mode)
            #   - count (number of unique items)
            #   - confidence(%) (average confidence score)
            #   - frame_presence(%) (percentage of frames with item)
            output_stats = tracker.get_output_stats()
            
            # Display table if we have detections
            if not output_stats.empty:
                st.dataframe(
                    output_stats, 
                    use_container_width=True  # Make table fill column width
                )
            else:
                # Show informative message if no items detected
                # This could mean low confidence threshold or no items in image
                st.info("🔍 No items detected.")

    except Exception as e:
        # =====================================================================
        # ERROR HANDLING
        # =====================================================================
        # Catch any errors during processing and display to user
        # Common errors:
        #   - Corrupt image file
        #   - Unsupported image format
        #   - Model inference failure
        #   - Memory issues with large images
        st.error(f"❌ Failed to process image: {e}")
        st.stop()  # Stop execution to prevent further errors