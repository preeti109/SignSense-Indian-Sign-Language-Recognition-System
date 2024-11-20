import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time
import logging
from datetime import datetime


logging.basicConfig(
    filename="gesturePredictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


model = load_model("trainedModel.keras")

class_names = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','None' ]

cap = cv2.VideoCapture(0)

roi_start_x, roi_start_y, roi_end_x, roi_end_y = 100, 100, 300, 300

fps = 30
frame_time = 1.0 / fps
prev_time = time.time()


def preprocess_image(frame, roi_start_x, roi_start_y, roi_end_x, roi_end_y):
    """Preprocesses the input image for model prediction."""
    cropped_frame = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
    img = cv2.resize(cropped_frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_gesture(model, img):
    """Makes prediction using the trained model."""
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    class_label = class_names[class_index]
    confidence = np.max(prediction)
    return class_label, class_index, confidence


frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()

    if current_time - prev_time >= frame_time:
        prev_time = current_time

        img = preprocess_image(frame, roi_start_x, roi_start_y, roi_end_x, roi_end_y)

        class_label, class_index, confidence = predict_gesture(model, img)

        logging.info(
            f"Predicted: {class_label} (Class Index: {class_index}), Confidence: {confidence:.2f}"
        )

        cv2.putText(
            frame,
            f"Label: {class_label} (Class: {class_index}) ({confidence:.2f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        frame, (roi_start_x, roi_start_y), (roi_end_x, roi_end_y), (0, 255, 0), 2
    )

    cv2.imshow("SignSence - Indian Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

logging.info(f"Exited program at {datetime.now()}")
