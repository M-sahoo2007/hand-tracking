# Code Citations

## License: unknown
https://github.com/RohanGodha/RockPaperScissors.git.io/tree/6e617f5c793659c5ae027eaa0b8e77961f9ff693/HandTrackingModule.py

```
import cv2
import mediapipe as mp
import time
import math
import numpy as np

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
```


## License: unknown
https://github.com/Cavin6080/Python/tree/f9b402801cae446161532f723bb8160c4880b131/Hand_Detection/hand_tracking_module.py

```
(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return
```


## License: unknown
https://github.com/T3CHNICK287/Volume-Gesture-Control/tree/40ace5e53cb7d5f530f27a3b4e9951ec451835f5/HandTrackingModule.py

```
.")
            return img

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img
```


## License: unknown
https://github.com/tohver/Portfolio/tree/5ecbc9dfc751b92ceee4c1bc6e8d2971be93724f/Computer%20Vision/Gesture%20Control/handTracking.py

```
.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y *
```


## License: unknown
https://github.com/vasiliyeskin/MachineLearningExperiences/tree/d6c1b1bccec9a1cef42cc5f0d603a1cd6b92c954/OpenCV/HandProjects/Project%201%20%E2%80%93%20Gesture%20Volume%20Control/HandTrackingModule.py

```
self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.
```

