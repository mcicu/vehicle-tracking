import os

import cv2
import numpy as np
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def mouse_callback(event, mouse_x, mouse_y, flags, canvas_state):
    if event == cv2.EVENT_LBUTTONDOWN and canvas_state["is_paused"]:
        canvas_state["is_drawing_on_frame"] = True
        canvas_state["wip_detection_area"].append((mouse_x, mouse_y))
        if len(canvas_state["wip_detection_area"]) == 4:
            cv2.polylines(
                canvas_state["annotated_frame"],
                [np.array(canvas_state["wip_detection_area"], np.int32)],
                True,
                (255, 0, 255),
                4,
            )
            cv2.imshow("YOLOv8 Tracking", canvas_state["annotated_frame"])
            canvas_state["detection_areas"].append(canvas_state["wip_detection_area"].copy())
            canvas_state["wip_detection_area"].clear()
            canvas_state["is_drawing_on_frame"] = False

    if event == cv2.EVENT_MOUSEMOVE and canvas_state["is_drawing_on_frame"]:
        temp_area = canvas_state["wip_detection_area"].copy()
        temp_area.append((mouse_x, mouse_y))
        temp_annotated_frame = canvas_state["annotated_frame"].copy()
        cv2.polylines(
            temp_annotated_frame,
            [np.array(temp_area, np.int32)],
            False,
            (255, 0, 255),
            4,
        )
        cv2.imshow("YOLOv8 Tracking", temp_annotated_frame)

    if event == cv2.EVENT_RBUTTONDOWN and canvas_state["is_paused"]:
        if canvas_state["is_drawing_on_frame"]:
            canvas_state["is_drawing_on_frame"] = False
            canvas_state["wip_detection_area"] = []
        elif len(canvas_state["detection_areas"]) > 0:
            canvas_state["detection_areas"].pop()

        canvas_state["annotated_frame"] = canvas_state["annotated_frame_no_user_areas"].copy()
        draw_user_areas(canvas_state["annotated_frame"], canvas_state["detection_areas"])
        cv2.imshow("YOLOv8 Tracking", canvas_state["annotated_frame"])


def draw_user_areas(frame, detection_areas):
    for detection_area in detection_areas:
        cv2.polylines(frame, [np.array(detection_area, np.int32)], True, (255, 0, 255), 4)


def main():
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Open the video file
    video_path = "radu-road.mp4"
    # video_path = "0911.mp4"
    # video_path = "road5sec.mp4"
    capture = cv2.VideoCapture(video_path)

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_count = 0
    detected_vehicles = dict()

    # canvas state parameters, shared with mouse callback function
    canvas_state = {
        "is_drawing_on_frame": False,
        "is_paused": False,
        "wip_detection_area": [],
        "detection_areas": [],
        "annotated_frame": None,
        "annotated_frame_no_user_areas": None
    }

    cv2.namedWindow("YOLOv8 Tracking")
    cv2.setMouseCallback(
        "YOLOv8 Tracking",
        mouse_callback,
        canvas_state,
    )

    # classes https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
    # 2 - car
    # 3 - motorcycle
    # 5 - bus
    # 7 - truck
    detected_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    detected_class_keys = [*detected_classes.keys()]
    detected_class_names = [*detected_classes.values()]

    # Loop through the video frames
    while capture.isOpened():
        canvas_state["is_paused"] = False
        canvas_state["wip_detection_area"] = []  # cleanup partial polygon
        # Read a frame from the video
        success, frame = capture.read()

        if not success:
            break  # end of video

        current_frame_count += 1
        if current_frame_count % 3 != 0:  # only analyse every Nth frame
            continue

        # Run YOLOv8 detection and tracking on the frame, persisting tracks between frames
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=detected_class_keys,
            tracker="bytetrack.yaml",
        )

        canvas_state["annotated_frame"] = frame
        if results[0].boxes.id is not None:
            # Visualize the detected bounding boxes on the frame
            canvas_state["annotated_frame"] = results[0].plot()

            # Process detected vehicles in the current frame, and save/update info about them
            for x1, y1, x2, y2, track_id, score, vehicle_type_id in results[0].boxes.data.numpy():
                # draw green dot on bottom right corner of bounding box
                cv2.circle(canvas_state["annotated_frame"], (int(x2), int(y2)), 6, (0, 255, 70), -1)

                # save new vehicle
                if detected_vehicles.get(track_id) is None:
                    detected_vehicles[track_id] = {
                        "track_id": track_id,
                        "score": score,
                        "vehicle_type": detected_classes[vehicle_type_id],
                        "detected_in_areas": set(),
                    }

                # update existing vehicle if score is better (vehicle type can change, depending on score confidence for the type)
                if detected_vehicles.get(track_id).get("score") < score:
                    detected_vehicles[track_id]["score"] = score
                    detected_vehicles[track_id]["vehicle_type"] = detected_classes[vehicle_type_id]

                # check if vehicle is in a drawn area, and save the polygon id (index)
                # note: we use (x2, y2) point (bottom right corner of bounding box) to check if it's in polygon
                for area_index, area in enumerate(canvas_state["detection_areas"]):
                    vehicle_in_area = cv2.pointPolygonTest(np.array(area, np.int32), (x2, y2), False)
                    if vehicle_in_area >= 0: # -1 if outside, 0 if on edge, 1 if inside of polygon
                        detected_vehicles[track_id]["detected_in_areas"].add(area_index)

        print("CURRENT FRAME = " + str(current_frame_count) + " / " + str(total_frames))
        print(detected_vehicles)

        # draw user areas (polygons)
        canvas_state["annotated_frame_no_user_areas"] = canvas_state["annotated_frame"].copy()
        draw_user_areas(canvas_state["annotated_frame"], canvas_state["detection_areas"])

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", canvas_state["annotated_frame"])

        k = cv2.waitKey(25) & 0xFF
        if k == 27 or k == ord("q") or k == ord("Q"):
            break
        if k == ord(" "):  # if spacebar is pressed
            canvas_state["is_paused"] = True
            paused_key = cv2.waitKey(0) & 0xFF  # program is paused for a while
            if paused_key == ord(" "):  # pressing space again unpauses the program
                pass

    # Release the video capture object and close the display window
    capture.release()
    cv2.destroyAllWindows()


# call main function
main()
