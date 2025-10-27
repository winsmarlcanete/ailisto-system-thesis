# app.py
import os
import time
from flask import Flask, render_template, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from camera import VideoCamera
from model_inference import Model
import cv2



app = Flask(__name__, instance_relative_config=True)

os.makedirs(app.instance_path, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.instance_path, 'ai-listo.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    label = db.Column(db.String(64))
    confidence = db.Column(db.Float)
    x1 = db.Column(db.Integer); y1 = db.Column(db.Integer)
    x2 = db.Column(db.Integer); y2 = db.Column(db.Integer)

# camera = VideoCamera(src=1)  # 0 = default webcam
model = Model(device='cpu')  # update device as needed

def draw_boxes(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = map(int, d['box'])
        label_txt = f"{d['label']}:{d['conf']:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label_txt, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame

def gen_frame():
    camera = cv2.VideoCapture(1)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def generate_frames():
#     while True:
#         frame = camera.get_frame()
#         if frame is None:
#             continue
#         detections = model.predict(frame)
#         # optional: store detections to DB (only store if detections exist)
#         if detections:
#             for d in detections:
#                 det = Detection(
#                     label=d['label'],
#                     confidence=float(d['conf']),
#                     x1=int(d['box'][0]), y1=int(d['box'][1]),
#                     x2=int(d['box'][2]), y2=int(d['box'][3])
#                 )
#                 db.session.add(det)
#             db.session.commit()
#         frame = draw_boxes(frame, detections)
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue
#         frame_bytes = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         if frame is None:
#             time.sleep(0.01)
#             continue
        
#         # Convert color after confirming frame exists
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         ret, jpeg = cv2.imencode('.jpg', frame_rgb)
#         if not ret:
#             time.sleep(0.01)
#             continue

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
#         time.sleep(0.03)  # ~30 fps


@app.route('/video_feed')
def video_feed():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def stats():
    rows = db.session.query(Detection.label, db.func.count(Detection.id)).group_by(Detection.label).all()
    data = {label: count for label, count in rows}
    return jsonify(data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
