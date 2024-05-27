import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.PoseLandmarkerOptions
VisionRunningMode= mp.tasks.vision.RunningMode

model_path = ''

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)
