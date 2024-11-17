from flask import Flask, Response, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import torch
import cv2
import threading

# Flask 애플리케이션 초기화
app = Flask(__name__)
CORS(app)  # CORS 활성화, 외부에서의 요청 허용
socketio = SocketIO(app, cors_allowed_origins="*")  # WebSocket CORS 설정

# YOLOv5 모델 로드 (객체 감지용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# MJPEG 비디오 스트림 설정 (Jetson Nano에서 처리한 비디오 스트림을 받아 처리)
cap = cv2.VideoCapture(0)

# 객체 감지 함수
def detect_objects(frame):
    results = model(frame)
    detections = []
    for *xyxy, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        confidence = conf.item()
        detections.append({'label': label, 'confidence': confidence})
    return detections

# MJPEG 비디오 스트림 생성
def generate_video_stream():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 객체 감지 수행
        detections = detect_objects(frame)
        
        # React 클라이언트에 객체 감지 정보 전송 (WebSocket)
        socketio.emit('detections', {'detections': detections})

        # MJPEG 형식으로 비디오 스트리밍
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# 비디오 스트림을 클라이언트로 전송하는 라우트
@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# API 엔드포인트 예시: 특정 데이터 제공
@app.route('/get_info', methods=['GET'])
def get_info():
    return jsonify({
        "message": "Jetson Nano에서 객체 감지 기능이 활성화되었습니다.",
        "status": "success"
    })

# WebSocket 연결을 통한 실시간 데이터 전송
@socketio.on('connect')
def handle_connect():
    print("클라이언트가 연결되었습니다.")
    emit('message', {'data': '클라이언트가 성공적으로 연결되었습니다!'})

# 클라이언트로부터 메시지 받기
@socketio.on('client_message')
def handle_client_message(message):
    print(f"클라이언트로부터 받은 메시지: {message['data']}")
    emit('server_response', {'data': '서버에서 받은 메시지 응답!'})

if __name__ == '__main__':
    # SSL 인증서 경로 설정 (HTTPS 설정)
    context = ('/path/to/cert.pem', '/path/to/key.pem')  # SSL 인증서 파일 경로
    # WebSocket과 API 서버 실행
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, ssl_context=context)
