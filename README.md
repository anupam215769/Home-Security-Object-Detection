# SafeHome Observer
SafeHome Observer is a security-focused project that integrates YOLO-based object detection and BLIP image captioning to identify and describe persons or vehicles within camera feeds or recorded footage. It’s designed to help users monitor and keep track of critical events in real time, making it suitable for deployments in home or office security systems.

## YOLO12 Detecting Person
![person](https://i.ibb.co/7J24S79W/person.png)

## Captioning For Person/Car
![results](https://i.ibb.co/cXsRFNpJ/results.png)


## Working Behind

1. Lets you upload a video (e.g., `sample.mp4`).
2. Reads it frame-by-frame with `OpenCV` (sampling every N frames to reduce processing time).
3. Performs `YOLO1`2 object detection on each sampled frame (using the `Ultralytics` library).
4. Generates a caption for each sampled frame using a `VisionEncoderDecoderModel`.
5. Prints the detected objects and caption text in the `Streamlit` app.


> Important:
- This script can be resource-intensive for larger/longer videos (especially on CPU).
- For best performance, run on a system (or cloud environment) that has a GPU with the correct PyTorch version.
- By default, it attempts to use GPU. If no GPU is available, it falls back to CPU.
