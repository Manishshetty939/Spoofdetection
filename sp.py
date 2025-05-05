import cv2
import numpy as np
import time

def detect_iris_region(gray_frame):
    circles = cv2.HoughCircles(
        gray_frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
        param1=100, param2=20, minRadius=20, maxRadius=80
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:  
            x, y, r = i
            return gray_frame[y - r:y + r, x - r:x + r]
    return None

cap = cv2.VideoCapture(0)
print("Starting Iris Spoof Detection...\nPlease position your eye close to the webcam.")

iris_textures = []
start_time = time.time()
timeout = 10  # in sec
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    iris = detect_iris_region(gray)

    if iris is not None and iris.size > 0:
        std = np.std(iris)
        iris_textures.append(std)
        frame_count += 1
        cv2.circle(frame, (gray.shape[1]//2, gray.shape[0]//2), 5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Iris not detected. Please move closer.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Iris Spoof Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break

    if frame_count >= 100 or (time.time() - start_time) > timeout:
        break

cap.release()
cv2.destroyAllWindows()

if len(iris_textures) == 0:
    print("\n Iris was not detected .  Try again with better lighting or camera angle.")
else:
    variation = np.std(iris_textures)
    print("\nAnalyzing texture variation...")
    print(f"Texture Std Deviation: {variation:.2f}")

    if variation > 5:
        print("Result: Likely a Real Eye.")
    else:
        print(" Result: Likely a Spoof (still image or printout).")
