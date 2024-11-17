from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# 카메라 연결 (0은 기본 카메라를 의미)
cap = cv2.VideoCapture(0)

# 카메라에서 프레임을 캡처하고, 클라이언트에게 전송할 수 있도록 JPEG 형식으로 변환
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 카메라에서 캡처한 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()

        # 클라이언트에게 보내는 스트리밍 형식으로 데이터를 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# /video 라우트에서 실시간 카메라 스트리밍 반환
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 기본 페이지 라우트 (웹 페이지)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
