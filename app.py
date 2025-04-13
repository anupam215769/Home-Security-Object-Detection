import streamlit as st
import cv2
import torch
import tempfile
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from datetime import datetime

###############################################################################
# YOLO Detection Function
###############################################################################

def detect_objects_yolo(model, frame_bgr):
    """
    Runs YOLO12 object detection on the given frame (BGR image).
    Returns a list of detected class labels, e.g. ["person", "car", ...].
    """
    # Convert BGR (OpenCV) to RGB (YOLO expects RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model.predict(frame_rgb, verbose=False)
    if not results:
        return []
    result = results[0]  # first result if multiple
    found_labels = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        found_labels.append(label)
    return found_labels

###############################################################################
# BLIP Model for Captioning
###############################################################################

class ImageCaptioner:
    """
    A wrapper around a BLIP model to generate short textual descriptions (captions).
    """

    def __init__(self, model_name="Salesforce/blip-image-captioning-large", device="cuda"):
        """
        model_name: String, any BLIP model name from Hugging Face (e.g. "Salesforce/blip-image-captioning-large")
        device: "cuda" to use GPU if available, else "cpu"
        """
        if not torch.cuda.is_available() or device.lower() == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        st.info(f"Loading BLIP model '{model_name}' on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate_caption(self, frame_bgr):
        """
        Generate a short caption for the BGR image (numpy array).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_length=16, min_length=5)
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

###############################################################################
# Streamlit App
###############################################################################

def main():
    st.title("SafeHome Observer")
    st.write("Conveys constant observation of a home environment, focusing on safeguarding against possible intrusions.")

    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        # 1. Save uploaded video to a temp file for OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            tmp_video_path = tmp_file.name

        # 2. Display the uploaded video for playback
        with open(tmp_video_path, "rb") as f:
            st.video(f.read(), format="video/mp4")

        # 3. Load YOLOv8 model
        st.info("Loading YOLO12 model, please wait...")
        yolo_model = YOLO("yolo12n.pt")  # downloads if necessary

        # 4. Load the captioning model (BLIP)
        captioner = ImageCaptioner(device="cuda")

        # 5. Open video with OpenCV
        cap = cv2.VideoCapture(tmp_video_path)
        if not cap.isOpened():
            st.error("Could not open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Video FPS: {fps:.2f}, Total frames: {total_frames}")

        # We sample every 30 frames => ~1 sample per second if 30 FPS
        sample_rate = 30
        frame_idx = 0

        results = []
        st.write("Analyzing video, please wait...")
        progress_bar = st.progress(0)

        # 6. Read each frame, sample, detect only if 'person' or 'car' is present
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            # Update progress
            progress_bar.progress(min(frame_idx / max(1, total_frames), 1.0))

            # Check if it's a frame we want to analyze
            if frame_idx % sample_rate == 0:
                detections = detect_objects_yolo(yolo_model, frame)
                # Convert list to set for easy checking
                detection_set = set(detections)
                # We only care about frames with 'person' or 'car'
                if "person" in detection_set or "car" in detection_set:
                    # If we found person/car, generate caption
                    caption_text = captioner.generate_caption(frame)

                    # Approximate timestamp in HH:MM:SS
                    time_sec = frame_idx / fps if fps > 0 else 0
                    time_str = datetime.utcfromtimestamp(time_sec).strftime("%H:%M:%S")

                    # Create a string of relevant objects
                    det_str = ", ".join(detections)

                    results.append((frame_idx, time_str, det_str, caption_text))

            frame_idx += 1

        cap.release()
        progress_bar.empty()

        st.subheader("Detected Person/Car Results")
        if not results:
            st.write("No frames with 'person' or 'car' were detected.")
        else:
            for (f_idx, ts, det_str, cap_text) in results:
                st.markdown(f"**Frame {f_idx} (approx {ts})**")
                st.write(f"- **Detections**: {det_str}")
                st.write(f"- **Caption**: {cap_text}")
                st.write("---")

        st.success("Done processing video.")

if __name__ == "__main__":
    main()
