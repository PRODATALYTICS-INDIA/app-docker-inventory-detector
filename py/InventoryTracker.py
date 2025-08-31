from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
from collections import defaultdict
import os

# -------------------------------------------------------------------------
# Load label catalog (Excel reference data)
# -------------------------------------------------------------------------

def load_label_catalog():
    """Load labelling-catalog dataframe (only once)."""
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "labelling-catalog.xlsx")
    return pd.read_excel(DATA_PATH)

label_catalog = load_label_catalog()
# build lookup dict for fast metadata access
sku_lookup = label_catalog.set_index("sku_code").to_dict(orient="index") 

# -------------------------------------------------------------------------
# YOLO tracker class
# -------------------------------------------------------------------------
class InventoryTracker:
    def __init__(self, model_path="models/model_25-08-02.pt", label_mode="item_name"):
        """
        Initializes the tracker with YOLO model and summary stats.
        
        Args:
            model_path (str): Path to YOLO model weights.
             label_mode (str): Label to display on frames and aggregate stats
                              ("sku_code", "item_name", "brand", "sub_category", "category").
        """
        self.model = YOLO(model_path) # YOLO model initialization
        self.confidence_threshold = 0.0
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        self.label_catalog = label_catalog
       
        # validate & assign label_mode
        self.valid_label_modes = {"sku_code", "item_name", "brand", "sub_category", "category"}
        if label_mode not in self.valid_label_modes:
            print(f"[WARN] Invalid label_mode '{label_mode}', falling back to 'item_name'")
            self.label_mode = "item_name"
        else:
            self.label_mode = label_mode

        self.reset_output_stats()

    def reset_output_stats(self):
        """Resets the statistics for a new video or session."""
        self.frame_count = 0
        self.class_appearances = defaultdict(int) # counts per sku
        self.overall_tracked_ids = defaultdict(set) # unique IDs per sku
        self.confidence = defaultdict(list) # list of probabilities per sku

    def track_picture_stream(self, frame: np.ndarray, confidence_threshold: float):
        """
        Core logic to process a single frame for detection, tracking, and stats gathering.

        Args:
            frame (np.ndarray): Input frame.
            confidence_threshold (float): YOLO confidence threshold.

        Returns:
            annotated_frame (np.ndarray): Frame with annotations.
            live_summary (dict): Running summary of detections.
        """
        self.frame_count += 1
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # annotation Logic
        labels = []
        for det in tracked_detections:
            class_id = det[3]
            tracker_id = det[4]
            # confidence = det[5] 
            confidence = det.conf if hasattr(det, 'conf') else results.boxes.conf[class_id]
            
            detected_sku = self.model.model.names[class_id] # e.g., "sku_1"
            meta = sku_lookup.get(detected_sku, {})
            
            # Extract confidence: supervision detections may store confidence in `det.conf` or results.boxes.conf
            if hasattr(det, 'conf'):
                confidence = det.conf
            else:
                # fallback: get confidence from YOLO results if available
                confidence = float(results.boxes.conf[class_id]) if len(results.boxes.conf) > class_id else 0.0

            # label to show on frame (controlled by self.label_mode)
            label_text = detected_sku if self.label_mode == "sku_code" else meta.get(self.label_mode, detected_sku)
            labels.append(f"#{tracker_id} {label_text}")
            
            # deduplication logic
            if tracker_id not in self.overall_tracked_ids[detected_sku]:
                self.overall_tracked_ids[detected_sku].add(tracker_id)
                self.class_appearances[detected_sku] += 1
                self.confidence[detected_sku].append(confidence)

        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
        annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=tracked_detections)

        live_summary = {name: len(ids) for name, ids in self.overall_tracked_ids.items() if len(ids) > 0}
        return annotated_frame, live_summary

    def get_output_stats(self):
        """
        Generates a summary DataFrame, aggregating by SKU or another attribute.
        
        Returns:
            pd.DataFrame: Aggregated summary.
        """
        if self.frame_count == 0:
            return pd.DataFrame()
            
        summary_data = []
        
        for sku, ids in self.overall_tracked_ids.items():
            if not ids:
                continue # if id set is empty it skips to the next SKU in the loop.
            meta = sku_lookup.get(sku, {})
            key_value = sku if self.label_mode == "sku_code" else meta.get(self.label_mode, sku)

            total_unique_items = len(ids)
            appearance_frames = self.class_appearances[sku]
            presence_percentage = (appearance_frames / self.frame_count) * 100
            mean_confidence = np.mean(self.confidence.get(sku, [0])) * 100
            
            summary_data.append({
                self.label_mode: key_value,
                "count": total_unique_items,
                "confidence(%)": f"{int(round(mean_confidence))}",
                "frame_presence(%)": f"{int(round(presence_percentage))}"
            })

        output = pd.DataFrame(summary_data)

        if self.label_mode != "sku_code" and not output.empty:
            output = output.groupby(self.label_mode, as_index=False).agg({
                "count": "sum",
                "confidence(%)": lambda x: f"{int(round(np.mean([float(v) for v in x])))}",
                "frame_presence(%)": lambda x: f"{int(round(np.mean([float(v) for v in x])))}"
            })

        return output

    def track_video_stream(self, frame_generator, confidence_threshold):
        """
        Processes frames from a video stream and yields results.

        Args:
            frame_generator (iterable): Yields frames.
            confidence_threshold (float): YOLO confidence threshold.

        Yields:
            (annotated_frame, live_summary)
        """
        self.reset_output_stats()
        for frame in frame_generator:
            yield self.track_picture_stream(frame, self.confidence_threshold)