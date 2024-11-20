import cv2
import os
from datetime import datetime


def initialize_capture(device_index=0):
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise IOError("Unable to access the webcam. Check the device connection.")
    return cap


def create_directory(gesture_name, base_dir="dataSet"):
    save_dir = os.path.join(base_dir, gesture_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def log_event(message, log_file="gesture_log.txt"):
    with open(log_file, "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] {message}\n")


def capture_gesture_images(
    gesture_name, max_images=200, roi_coords=(100, 100, 300, 300), base_dir="dataSet"
):
    save_dir = create_directory(gesture_name, base_dir)
    cap = initialize_capture()
    image_count = 0

    print(f"Starting collection for gesture: '{gesture_name}'")
    print("Ensure your hand is placed within the green rectangle.")
    print("Press 'q' at any time to exit the collection process.")

    log_event(f"Started collecting images for gesture '{gesture_name}'.")

    while image_count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            log_event("Webcam read error encountered. Exiting.")
            break

        x1, y1, x2, y2 = roi_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cropped_frame = frame[y1:y2, x1:x2]
        cv2.imshow("Gesture Capture", frame)

        file_path = os.path.join(save_dir, f"{gesture_name}_{image_count:03d}.jpg")
        cv2.imwrite(file_path, cropped_frame)
        image_count += 1
        print(f"Image {image_count}/{max_images} saved.")

        log_event(f"Image {image_count} saved to {file_path}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting on user request.")
            log_event("User terminated the collection process early.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if image_count == max_images:
        print(
            f"Successfully collected {max_images} images for gesture '{gesture_name}'."
        )
        log_event(
            f"Completed collection of {max_images} images for gesture '{gesture_name}'."
        )
    else:
        print(f"Collection terminated after {image_count} images.")
        log_event(
            f"Terminated collection after {image_count} images for gesture '{gesture_name}'."
        )


def main():
    gesture_name = input("Enter the gesture name to collect images for: ").strip()
    if not gesture_name:
        print("Error: Gesture name cannot be empty.")
        return

    max_images = int(
        input("Enter the number of images to capture (default 200): ") or 200
    )
    roi_coords = (100, 100, 300, 300)

    print("\n--- Gesture Image Collection ---")
    print(f"Gesture Name: {gesture_name}")
    print(f"Number of Images: {max_images}")
    print(f"ROI Coordinates: {roi_coords}")
    print("\nStarting in 3 seconds...")
    cv2.waitKey(3000)

    try:
        capture_gesture_images(gesture_name, max_images, roi_coords)
    except Exception as e:
        print(f"An error occurred: {e}")
        log_event(f"Error: {e}")


if __name__ == "__main__":
    main()
