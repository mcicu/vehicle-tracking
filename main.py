import os
import csv

import cv2
import numpy as np
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# vehicle types https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
# 2 - car
# 3 - motorcycle
# 5 - bus
# 7 - truck
vehicle_types = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
vehicle_type_keys = [*vehicle_types.keys()]
vehicle_type_names = [*vehicle_types.values()]


def mouse_callback(event, mouse_x, mouse_y, flags, canvas_state):
    if event == cv2.EVENT_LBUTTONDOWN and canvas_state["is_paused"]:
        canvas_state["is_drawing_on_frame"] = True
        canvas_state["wip_detection_area"].append((mouse_x, mouse_y))
        if len(canvas_state["wip_detection_area"]) == 4:
            # compute center of area
            area_moments = cv2.moments(np.int32(canvas_state["wip_detection_area"]))
            center_x = int(area_moments["m10"] / area_moments["m00"])
            center_y = int(area_moments["m01"] / area_moments["m00"])
            # update canvas_state
            canvas_state["detection_areas"].append({
                "points": canvas_state["wip_detection_area"].copy(),
                "center": [center_x, center_y]})
            canvas_state["wip_detection_area"].clear()
            canvas_state["is_drawing_on_frame"] = False
            # draw new area
            draw_user_areas(canvas_state["annotated_frame"], canvas_state["detection_areas"])
            cv2.imshow("YOLOv8 Tracking", canvas_state["annotated_frame"])

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
    for area_index, area in enumerate(detection_areas):
        cv2.polylines(frame, [np.array(area["points"], np.int32)], True, (255, 0, 255), 4)
        cv2.putText(
            frame,
            "Area " + str(area_index),
            (area["center"][0]-25, area["center"][1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )


def compute_results_and_write_to_file(detected_vehicles, filename):
    result_per_direction = {}
    for detected_vehicle in detected_vehicles.values():
        # we take the direction to be the areas through which the vehicle goes (area ordering matters)
        direction = "direction-" + "-".join(map(str, detected_vehicle["detected_in_areas"].keys()))
        vehicle_type = detected_vehicle["vehicle_type"]
        if result_per_direction.get(direction) is None:
            vehicles_types_count_0_dictionary = {k:0 for k in vehicle_type_names}
            result_per_direction[direction] = {"direction": direction, "total": 0} | vehicles_types_count_0_dictionary
        result_per_direction[direction][vehicle_type] += 1
        result_per_direction[direction]["total"] += 1

    with open(filename, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["direction", "total"] + vehicle_type_names)
        writer.writeheader()
        # write results sorted by vehicle total (descending)
        for k, v in sorted(result_per_direction.items(), key=lambda item: item[1]["total"], reverse=True):
            writer.writerow(v)


def main():
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Open the video file
    video_path = "road.mp4"
    capture = cv2.VideoCapture(video_path)

    video_path_without_extension = os.path.splitext(video_path)[0]
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_count = 0
    frame_step = 2
    file_output_frame_step = 5000
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

    # Loop through the video frames
    while capture.isOpened():
        canvas_state["is_paused"] = False
        canvas_state["wip_detection_area"] = []  # cleanup partial polygon
        # Read a frame from the video
        success, frame = capture.read()

        # end of video, write final results to csv
        if not success:
            # delete partial results (if exists)
            if os.path.exists(video_path_without_extension + ".partial.csv"):
                os.remove(video_path_without_extension + ".partial.csv")
            filename = video_path_without_extension + ".final.csv"
            compute_results_and_write_to_file(detected_vehicles, filename)
            break

        # only analyse every Nth frame (based on frame_step)
        current_frame_count += 1
        if current_frame_count % frame_step != 0:
            continue

        # write partial results to csv file, after every Nth frame (based on file_output_frame_step)
        if current_frame_count % (file_output_frame_step - file_output_frame_step % frame_step) == 0:
            filename = video_path_without_extension + ".partial.csv"
            compute_results_and_write_to_file(detected_vehicles, filename)

        print("CURRENT FRAME = " + str(current_frame_count) + " / " + str(total_frames))

        # Run YOLOv8 detection and tracking on the frame, persisting tracks between frames
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=vehicle_type_keys,
            tracker="bytetrack.yaml",
        )

        canvas_state["annotated_frame"] = frame
        if results[0].boxes.id is not None:
            # Visualize the detected bounding boxes on the frame
            canvas_state["annotated_frame"] = results[0].plot()

            # Process detected vehicles in the current frame, and save/update info about them
            for x1, y1, x2, y2, track_id, score, vehicle_type_id in results[0].boxes.data.numpy():
                # draw green dot on bottom right corner of bounding box
                cv2.circle(canvas_state["annotated_frame"], (int(x2), int(y2)), 8, (0, 255, 70), -1)

                # save new vehicle
                if detected_vehicles.get(track_id) is None:
                    detected_vehicles[track_id] = {
                        "track_id": track_id,
                        "score": score,
                        "vehicle_type": vehicle_types[vehicle_type_id],
                        "detected_in_areas": {},  # we use dictionary to keep insertion order (set doesn't preserve insertion order)
                    }

                # update existing vehicle if score is better (vehicle type can change, depending on score confidence for the type)
                if detected_vehicles.get(track_id).get("score") < score:
                    detected_vehicles[track_id]["score"] = score
                    detected_vehicles[track_id]["vehicle_type"] = vehicle_types[vehicle_type_id]

                # check if vehicle is in a drawn area, and save the polygon id (index)
                # note: we use (x2, y2) point (bottom right corner of bounding box) to check if it's in polygon
                for area_index, area in enumerate(canvas_state["detection_areas"]):
                    vehicle_in_area = cv2.pointPolygonTest(np.array(area["points"], np.int32), (x2, y2), False)
                    if vehicle_in_area >= 0:  # -1 if outside, 0 if on edge, 1 if inside of polygon
                        detected_vehicles[track_id]["detected_in_areas"][area_index] = area_index

                print(detected_vehicles[track_id])

        # draw user areas (polygons)
        canvas_state["annotated_frame_no_user_areas"] = canvas_state["annotated_frame"].copy()
        draw_user_areas(canvas_state["annotated_frame"], canvas_state["detection_areas"])

        # Prepare to display the annotated frame
        cv2.imshow("YOLOv8 Tracking", canvas_state["annotated_frame"])

        # Render the frame on screen (!) and pause for 35ms to capture any keyboard key
        k = cv2.waitKey(35) & 0xFF
        if k == 27 or k == ord("q") or k == ord("Q"):
            break
        if k == ord(" "):  # pause if spacebar is pressed
            canvas_state["is_paused"] = True
            # compute partial results on pause
            filename = video_path_without_extension + ".partial.csv"
            compute_results_and_write_to_file(detected_vehicles, filename)

            # program is paused indefinitely until a key is pressed
            cv2.waitKey(0) & 0xFF

    # Release the video capture object and close the display window
    capture.release()
    cv2.destroyAllWindows()


# call main function
main()
