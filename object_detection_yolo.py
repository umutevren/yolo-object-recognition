import cv2
import numpy as np
import os

# Load YOLO
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    # Adjusting the output layer extraction to handle different formats
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers

# Load COCO class labels
def load_coco_labels():
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Perform object detection
def detect_objects(img, net, output_layers):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
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

    return boxes, confidences, class_ids

# Draw bounding boxes on the image
def draw_labels(img, boxes, confidences, class_ids, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

# Process a single image
def process_image(img_path, net, output_layers, classes, output_dir):
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
    
    # Detect objects
    boxes, confidences, class_ids = detect_objects(img, net, output_layers)
    
    # Draw labels
    draw_labels(img, boxes, confidences, class_ids, classes)
    
    # Save the processed image
    output_path = os.path.join(output_dir, f"detected_{os.path.basename(img_path)}")
    cv2.imwrite(output_path, img)
    print(f"Saved detection result to: {output_path}")
    
    return img

# Main function to run the object detection
def main():
    # Create output directory if it doesn't exist
    output_dir = "detection_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load YOLO model and classes
    net, output_layers = load_yolo()
    classes = load_coco_labels()

    # List of images to process
    image_paths = ["image2.jpg", "image3.jpg", "image4.jpg"]

    # Process each image
    for img_path in image_paths:
        print(f"Processing {img_path}...")
        img = process_image(img_path, net, output_layers, classes, output_dir)
        
        if img is not None:
            # Display the image
            cv2.imshow(f"Detection Result - {img_path}", img)
            print(f"Press any key to continue to next image...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
