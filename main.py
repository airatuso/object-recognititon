from imageai.Detection import ObjectDetection
import cv2
import time

camera = cv2.VideoCapture("Desktop/Street.mp4")

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel()

finish = 0

while camera.isOpened():
    ret, frame = camera.read()
    
    start = time.time()
    if start - finish >2:
        _, array_detection = detector.detectObjectsFromImage(input_image=frame, input_type="array", output_type="array")
        finish = time.time()
        print(array_detection)

    cv2.imshow('Test', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()