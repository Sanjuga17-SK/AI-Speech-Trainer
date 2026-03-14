import cv2
import numpy as np
from deepface import DeepFace
from agno.tools import tool
import json

def log_before_call(fc):
    """Pre-hook function that runs before the tool execution"""
    print(f"About to call function with arguments: {fc.arguments}")

def log_after_call(fc):
    """Post-hook function that runs after the tool execution"""
    print(f"Function call completed with result: {fc.result}")

@tool(
    name="analyze_facial_expressions",
    description="Analyzes facial expressions to detect emotions and engagement using DeepFace and OpenCV.",
    show_result=True,
    stop_after_tool_call=True,
    pre_hook=log_before_call,
    post_hook=log_after_call
)
def analyze_facial_expressions(video_path: str) -> str:
    """
    Analyzes facial expressions in a video to detect emotions and engagement.

    Args:
        video_path: The path to the video file.

    Returns:
        A JSON string containing the emotion timeline and engagement metrics.
    """
    # Initialize OpenCV Haar Cascades for eye detection as a fallback for MediaPipe
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return json.dumps({"error": f"Could not open video at {video_path}"})

    emotion_timeline = []
    eye_contact_count = 0
    smile_count = 0
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30 # Default

    # Process every nth frame for performance optimization (approx 1 frame per second)
    # Most videos are ~30fps
    frame_interval = int(fps) if fps > 0 else 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Emotion Detection using DeepFace
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if analysis:
                dominant_emotion = analysis[0]['dominant_emotion']
                
                # Check for smile via emotion
                if dominant_emotion == "happy":
                    smile_count += 1

                timestamp = round(frame_count / fps, 2)
                emotion_timeline.append({"timestamp": timestamp, "emotion": dominant_emotion})
                
                # Eye Contact Detection using Haar Cascades
                # If we detect both eyes, we assume "eye contact" for this segment
                eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
                if len(eyes) >= 2:
                    eye_contact_count += 1

        except Exception as e:
            print(f"Error analyzing frame at {frame_count}: {e}")
            continue

    cap.release()

    # Calculate frequencies based on total analyzed frames
    total_analyzed = len(emotion_timeline) if emotion_timeline else 1
    
    engagement_metrics = {
        "eye_contact_frequency": round(eye_contact_count / total_analyzed, 2),
        "smile_frequency": round(smile_count / total_analyzed, 2)
    }

    return json.dumps({
        "emotion_timeline": emotion_timeline,
        "engagement_metrics": engagement_metrics
    })