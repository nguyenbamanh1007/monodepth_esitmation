from flask import Flask, request, jsonify, render_template
import base64
from PIL import Image
import io
import numpy as np
import cv2
import os
import math

# Import the necessary functions from run.py
from run import process_image, calculate_projection

app = Flask(__name__)
# Đọc ma trận độ sâu từ file .txt
depth_matrix = np.loadtxt('/home/manh/WED_distance_calculator/ma_tran_depth.txt')
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'  # New folder for saving output images

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):  # Create the output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.get_json()
    image_data = data['image']
    points = data['points']

    # Decode the image
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Save the image to a file
    image_path = os.path.join(UPLOAD_FOLDER, 'input_image.png')
    image.save(image_path)

    # Convert the image path to a string
    image_path_str = str(image_path)

    # Process image with the model
    output_image_np, distance = process_image_with_model(image_path_str, points)

    # Save the output image to the specified folder
    output_image_path = os.path.join(OUTPUT_FOLDER, 'output_image.png')
    output_pil_image = Image.fromarray(output_image_np)
    output_pil_image.save(output_image_path)

    # Convert the output image back to base64
    buffered = io.BytesIO()
    output_pil_image.save(buffered, format="PNG")
    output_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Định dạng giá trị distance với hai chữ số sau dấu chấm
    formatted_distance = round(distance, 2)

    return jsonify({"output_image": "data:image/png;base64," + output_base64, "distance": formatted_distance})

def process_image_with_model(image_path, points):
    # Run your model
    output_image = process_image(image_path)

    # Optionally, add points and lines on the output image
    output_image_np = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # Assuming output_image is in BGR format
    pt1 = (int(points[0]['x']), int(points[0]['y']))
    pt2 = (int(points[1]['x']), int(points[1]['y']))
    cv2.circle(output_image_np, pt1, 5, (0, 255, 0), -1)
    cv2.circle(output_image_np, pt2, 5, (0, 255, 0), -1)
    cv2.line(output_image_np, pt1, pt2, (0, 255, 0), 2)

    # Chuyển đổi tọa độ điểm nhấp chuột thành tọa độ ma trận
    pt1 = points[0]
    pt2 = points[1]
    z1 = depth_matrix[int(pt1['y']), int(pt1['x'])]
    z2 = depth_matrix[int(pt2['y']), int(pt2['x'])]

    center = {'x': 320, 'y': 240}
    width = 640
    height = 480
    projection_factor = {'x': 1.1178, 'y': 1.0238}
    x1, y1 = calculate_projection(z1, projection_factor, points[0], center, width, height)
    x2, y2 = calculate_projection(z2, projection_factor, points[1], center, width, height)

    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    print("Giá trị distance là:", distance)

    return output_image_np, distance

if __name__ == '__main__':
    app.run(debug=True)
