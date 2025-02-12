import streamlit as st
import io
from typing import Any
import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from PIL import Image

class Inference:
    def __init__(self, model_path="best.pt", **kwargs: Any):
        """
        Initializes the Inference class with the model set to best.pt.

        Args:
            model_path (str): Path to the model file (default is "best.pt").
        """
        check_requirements("streamlit>=1.29.0")  # Check Streamlit version
        self.st = st  # Reference to Streamlit instance
        self.source = None  # Placeholder for video source details
        self.enable_trk = False  # Flag to toggle object tracking
        self.conf = 0.25  # Confidence threshold for detection
        self.iou = 0.45  # IoU threshold for non-maximum suppression
        self.org_frame = None  # Original frame container
        self.ann_frame = None  # Annotated frame container
        self.vid_file_name = None  # Holds the name of the video file
        self.img_file_name = None  # Holds the name of the image file
        self.selected_ind = []  # List of selected classes
        self.model = None  # Loaded model instance
        self.temp_dict = {"model": model_path, **kwargs}
        self.model_path = model_path  # Set the model path to "best.pt" by default
        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Sets up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """
        <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .title {color: #FF64DA; text-align: center; font-size: 40px; margin-top: -50px; font-family: 'Archivo', sans-serif; margin-bottom: 20px;}
            .subtitle {color: #042AFF; text-align: center; font-size: 18px; font-family: 'Archivo', sans-serif; margin-top: -15px; margin-bottom: 50px;}
            .app-background {background-color: #f0f4f8;}
            .btn {background-color: #FF64DA; color: white; border-radius: 10px; padding: 10px 20px; font-size: 16px; cursor: pointer;}
            .btn:hover {background-color: #FF48C1;}
        </style>
        """
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)

        self.st.markdown('<div class="title">Byte-Bots</div>', unsafe_allow_html=True)
        self.st.markdown('<div class="subtitle">Underwater Sea-Waste Detection System</div>', unsafe_allow_html=True)

    def sidebar(self):
        """Configures the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:  # Add Ultralytics LOGO
            logo = "logo.png"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")  # Add elements to vertical setting menu
        self.source = self.st.sidebar.radio("Choose Input", ("Upload Video", "Upload Image"))  # Choose input type
        self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))  # Enable object tracking
        self.conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
        )  # Slider for confidence
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))  # Slider for NMS threshold

        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

    def source_upload(self):
        """Handles video and image uploads through the Streamlit interface."""
        if self.source == "Upload Video":
            self.vid_file_name = ""
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # BytesIO Object
                with open("ultralytics.mp4", "wb") as out:  # Open temporary file as bytes
                    out.write(g.read())  # Read bytes into file
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "Upload Image":
            self.img_file_name = ""
            img_file = self.st.sidebar.file_uploader("Upload Image File", type=["jpg", "png", "jpeg"])
            if img_file is not None:
                image = Image.open(img_file)
                image.save("input_image.jpg")
                self.img_file_name = "input_image.jpg"

    def configure(self):
        """Configures the model (hardcoded to best.pt) and loads selected classes for inference."""
        self.model = YOLO(self.model_path)  # Load the YOLO model (hardcoded to best.pt)
        class_names = list(self.model.names.values())  # Convert dictionary to list of class names
        self.st.success("Model loaded successfully!")

        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]
        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        """Performs real-time object detection inference."""
        self.web_ui()  # Initialize the web interface
        self.sidebar()  # Create the sidebar
        self.source_upload()  # Upload the source (video or image)
        self.configure()  # Configure the app

        if self.st.sidebar.button("Start", key="start_btn"):
            stop_button = self.st.button("Stop", key="stop_btn")  # Button to stop the inference
            
            if self.vid_file_name:  # If video is uploaded
                cap = cv2.VideoCapture(self.vid_file_name)  # Capture the video
                if not cap.isOpened():
                    self.st.error("Could not open video.")
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        self.st.warning("Failed to read frame from video. Please verify the video file.")
                        break

                    if self.enable_trk == "Yes":
                        results = self.model.track(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True)
                    else:
                        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                    annotated_frame = results[0].plot()  # Add annotations on frame

                    if stop_button:
                        cap.release()  # Release the capture
                        self.st.stop()  # Stop Streamlit app

                    self.org_frame.image(frame, channels="BGR")  # Display original frame
                    self.ann_frame.image(annotated_frame, channels="BGR")  # Display processed frame

                cap.release()  # Release the capture

            elif self.img_file_name:  # If image is uploaded
                img = cv2.imread(self.img_file_name)  # Read the uploaded image
                results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_img = results[0].plot()  # Annotate the image

                if stop_button:
                    self.st.stop()  # Stop Streamlit app

                self.org_frame.image(img, channels="BGR")  # Display original image
                self.ann_frame.image(annotated_img, channels="BGR")  # Display annotated image

        cv2.destroyAllWindows()  # Destroy window


if __name__ == "__main__":
    inf = Inference()  # Instantiating the Inference class with the default 'best.pt'
    inf.inference()  # Start the inference process
