from flask import Flask, render_template, Response, jsonify
import cv2
import threading

app = Flask(__name__)

# 카메라 연결
cap = cv2.VideoCapture(0)

# YOLO 모델 파일 경로
yolo_net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i-1] for i in yolo_net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def get_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    height, width, channels = frame.shape
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 50% 이상일 때만 인식
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return class_ids, boxes, indexes

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        class_ids, boxes, indexes = get_objects(frame)

        # 차량 클래스 (자동차, 트럭, 버스 등)만 필터링
        vehicle_types = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if label in ["car", "truck", "bus"]:
                    vehicle_types.append(label)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 차량 정보를 클라이언트에 보내기
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vehicle_info')
def vehicle_info():
    # 차량 종류를 JSON 형식으로 반환 (예시)
    vehicle_types = ["car", "truck"]
    return jsonify(vehicle_types)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
