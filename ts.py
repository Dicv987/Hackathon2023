from flask import Flask, redirect, render_template, Response, request, url_for
import cv2
import imutils
import time
import mediapipe as mp
import pyshine as ps
import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def pyshine_process():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
    classes = ["Weapon"]
    fps = 0
    st = 0
    frames_to_count = 20
    cnt = 0
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in np.squeeze(net.getUnconnectedOutLayers())]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(1)
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            height, width, _ = frame.shape
            print(frame[0, 0])
            print(height, width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            # Showing information on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.1:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            # Toda accion que se haga mientras tenga detectada la cara será dentro de este for
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            k = cv2.waitKey(30)

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks is not None:
                for hand_lankmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_lankmarks, mp_hands.HAND_CONNECTIONS)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            if indexes == 0:print("weapon detected in frame")
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
            if cnt == frames_to_count:
                try:
                    fps = round(frames_to_count / (time.time() - st))
                    st = time.time()
                    cnt = 0
                except Exception:
                    pass
            cnt += 1
            frame = imutils.resize(frame, width=1000)
            text = str(time.strftime("%d %b %Y %H.%M.%S %p"))
            frame = ps.putBText(frame, text, text_offset_x=190,
                                text_offset_y=30, background_RGB=(228, 20, 222))
            frame = cv2.imencode('.JPEG', frame, [cv2.IMWRITE_JPEG_QUALITY, 20])[1].tobytes()
            time.sleep(0.016)

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/results')
def video_feed():
    return Response(pyshine_process(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True, host="192.168.1.15", port=8080, threaded=True)