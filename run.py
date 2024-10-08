
import cv2
import math
from yoloseg import YOLOSeg

def calculate_projection(z, projection_factor, pixel, center, width, height):
    x = z * projection_factor['x'] * (pixel['x'] - center['x']) / width
    y = z * projection_factor['y'] * (pixel['y'] - center['y']) / height
    return x, y

def process_image(image_path):
    # Placeholder for the actual model processing
    # This function should process the image and return the output image
    model_path = "/home/manh/WED_distance_calculator/yolov8m-seg.onnx"
    yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)

    image = cv2.imread(image_path)
    img = cv2.imread(image_path)

    # Detect Objects
    boxes, scores, class_ids, masks = yoloseg(img)

    # Draw detections
    output_image = yoloseg.draw_masks(img)

    return output_image
