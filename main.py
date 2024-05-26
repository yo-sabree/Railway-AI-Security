import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
from flask import Flask, render_template, Response

app = Flask(__name__)

model = YOLO('yolov8s.pt')

# Dictionary to store objects_rect and class_list for each scenario
scenarios_data = {

    'crowd_management': {'objects_rect': [], 'class_list': []},
    'restricted_area_entry': {'objects_rect': [], 'class_list': []},
    'suitcase_handbag_detection': {'objects_rect': [], 'class_list': []},
    'work_monitoring': {'objects_rect': [], 'class_list': []}
}

def count_objects(scenario):
    count = 0
    for x1, y1, x2, y2, class_id in scenarios_data[scenario]['objects_rect']:
        c = scenarios_data[scenario]['class_list'][class_id]
        if scenario == 'crowd_management':
            if 'person' in c:
                count += 1
        elif scenario == 'restricted_area_entry':
            if 'person' in c:
                count += 1
        elif scenario == 'suitcase_handbag_detection':
            if 'suitcase' in c or 'handbag' in c or 'backpack' in c:
                count += 1
        elif scenario == 'work_monitoring':
            if 'person' in c:
                count += 1
    return count

def draw_count(frame, count,scenario):
    y_offset = 30
    if scenario == "suitcase_handbag_detection":
        for cls, cls_count in count.items():
            text = f"Bag: {cls_count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        for cls, cls_count in count.items():
            if cls_count > 0:
                text = f"{cls.capitalize()}: {cls_count}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_offset += 30

# Initialize the Flask app
app = Flask(__name__)

# Update the video sources
caps = {
    'crowd_management': cv2.VideoCapture('video01.mp4'),
    'restricted_area_entry': cv2.VideoCapture('Crime02.mp4'),
    'suitcase_handbag_detection': cv2.VideoCapture('Crime04.mp4'),
    'work_monitoring': cv2.VideoCapture('Work01.mp4')
}

# Read the COCO class list
with open("coco.txt", "r") as file:
    data = file.read()
    for scenario in scenarios_data:
        scenarios_data[scenario]['class_list'] = data.split("\n")

tracker = Tracker()

@app.route('/')
def index():
    return render_template('index.html')

def generate(scenario):
    while True:
        ret, frame = caps[scenario].read()
        if not ret:
            break

        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        scenarios_data[scenario]['objects_rect'] = []

        for index, row in px.iterrows():
            x1, y1, x2, y2, class_id = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
            c = scenarios_data[scenario]['class_list'][class_id]

            if scenario == 'crowd_management':
                if 'person' in c:
                    scenarios_data[scenario]['objects_rect'].append([x1, y1, x2, y2, class_id])
            elif scenario == 'restricted_area_entry':
                if 'person' in c:
                    scenarios_data[scenario]['objects_rect'].append([x1, y1, x2, y2, class_id])
            elif scenario == 'suitcase_handbag_detection':
                if 'suitcase' in c or 'handbag' in c or 'backpack' in c:
                    scenarios_data[scenario]['objects_rect'].append([x1, y1, x2, y2, class_id])
            elif scenario == 'work_monitoring':
                if 'person' in c:
                    scenarios_data[scenario]['objects_rect'].append([x1, y1, x2, y2, class_id])

        count = count_objects(scenario)

        for bbox in scenarios_data[scenario]['objects_rect']:
            x1, y1, x2, y2, _ = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        draw_count(frame, {'Person': count},scenario)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<scenario>')
def video_feed(scenario):
    return Response(generate(scenario), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_object_count/<scenario>')
def get_object_count(scenario):
    count = count_objects(scenario)
    return {'count': count}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
