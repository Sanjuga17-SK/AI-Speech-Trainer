import mediapipe as mp
try:
    print(f"MediaPipe version: {mp.__version__}")
    print(f"Solutions: {mp.solutions}")
    print("Success!")
except AttributeError as e:
    print(f"AttributeError: {e}")
    import mediapipe.solutions
    print(f"After explicit import, solutions: {mp.solutions}")
