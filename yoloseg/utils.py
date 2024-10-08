import numpy as np
import cv2

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
data = np.loadtxt('/home/manh/WED_distance_calculator/912_color.txt')
new_height = 392
new_width = 518
# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):
    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    mask_img = draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)

    
    for i in range(len(boxes)): 
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        color = colors[class_id]
        # depth = caculate_depth(i, image, boxes, class_ids, mask_alpha, mask_maps)
        extracted_array = mask_maps[i]
        np.savetxt('extracted_array.txt', extracted_array, fmt='%d')
        mask = cv2.resize(extracted_array, (new_width, new_height))
        np.savetxt('mask.txt', mask, fmt='%d')
        def apply_mask(data, mask):
            result = np.copy(data)
            result[mask == 0] = 0
            return result
        result_matrix = apply_mask(data, mask)

        
        non_zero_values = result_matrix[result_matrix != 0]
        mean_value = np.mean(non_zero_values)
        depth = int(mean_value * 100) / 100
        
        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        label = class_names[class_id]
        print("label and depth:",label,depth)
        caption = f'{label} {int(score * 100)}% -:{depth} m'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return mask_img



def draw_masks(image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for i in range(len(boxes)): 
        box = boxes[i]
        class_id = class_ids[i]
        color = colors[class_id]
        # if i >=3:  # Chỉ chạy 2 lần
        #     break
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img
        # Calculate center coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
       
        print("Center coordinates of circle:", (center_x, center_y))
        # Draw a red circle at the center
        # Draw line connecting center and top-left corner of the bounding box
        
        # cv2.circle(mask_img, (center_x, center_y), 5, (0, 0, 255), -1)

        # for j in range( 1, 2, 1):
        #     next_box = boxes[j]
        #     next_center_x = int((next_box[0] + next_box[2]) / 2)
        #     next_center_y = int((next_box[1] + next_box[3]) / 2)
        #     cv2.line(mask_img, (center_x, center_y), (next_center_x, next_center_y), (0, 0, 255), 2)
    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

def caculate_depth(i, image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        extracted_array = mask_maps[i]
        mask = np.resize(extracted_array, (new_height, new_width))
        def apply_mask(data, mask):
            result = np.copy(data)
            result[mask == 0] = 0
            return result
    result_matrix = apply_mask(data, mask)
    non_zero_values = result_matrix[result_matrix != 0]
    mean_value = np.mean(non_zero_values)
    mean_value = int(mean_value * 1000) / 1000
    return mean_value

