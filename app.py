from flask import Flask, request, render_template, send_from_directory
import cv2
import os
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)


# Load YOLO model
model_path = 'yolo_model/best.pt'  # Ensure the path is correct and consistent
model = YOLO(model_path)

def predict_and_save_image(path_test_car, output_image_path):
    try:
        results = model.predict(path_test_car, device='cpu')
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image)
        return output_image_path
    except Exception as e:
        return None


def predict_and_plot_video(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu')
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            out.write(frame)
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        return None


def process_media(input_path, output_path):
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        return predict_and_plot_video(input_path, output_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_path, output_path)
    else:
        return None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            input_path = os.path.join('uploads', file.filename)
            output_path = os.path.join('uploads', f'output_{file.filename}')
            os.makedirs('uploads', exist_ok=True)
            file.save(input_path)
            result_path = process_media(input_path, output_path)
            if result_path:
                return send_from_directory('uploads', os.path.basename(result_path))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
