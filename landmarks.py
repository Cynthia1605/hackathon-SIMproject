"""MediaPipe Pose (33-landmark) index constants shared by all pipeline modules.

Kept free of any mediapipe import so analysis code and tests run without it.
"""

NUM_LANDMARKS = 33

NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Channel layout of the cached 2D arrays: (num_frames, 33, 3)
X, Y, VIS = 0, 1, 2
