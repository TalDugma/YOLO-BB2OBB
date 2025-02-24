#  Code to generate OBB labels from BB labels utilizing sam2
import utils
import numpy as np
from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from collections import defaultdict
import ast
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path


# define new class "annotations"
class Annotations():
    """
    has the following attributes:
    - image_file: str
    - bounding_boxes: list of lists
    - image: np.array
    - mask: np.arrays
    """
    def __init__(self, image_file, bounding_boxes, image):
        self.image_file = image_file
        self.bounding_boxes = bounding_boxes
        self.image = image
        self.masks = []
        self.obb = []
    
    def __str__(self):
        return f"Annotations(image_file={self.image_file}, bounding_boxes={self.bounding_boxes}, image={self.image})"
    





# Check device availability and configure PyTorch
def configure_device(dev):
    if torch.cuda.is_available():
        device = torch.device(dev)
        torch.autocast(dev, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device
    else:
        return torch.device("cpu")  # Fallback to CPU if no GPUs are available

def initialize_predictor(model_cfg, checkpoint_path, device):
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    return SAM2ImagePredictor(sam2_model)


def generate_mask(bbox, frame, predictor):
    predictor.set_image(frame)
    cls, x, y, w, h = bbox
    x1, y1, x2, y2 = utils.xywh2xyxy(x, y, w, h)
    x1, y1, x2, y2 = utils.denormallize_coords(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
    input_box = np.array([x1, y1, x2, y2])
    masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
) 
    mask = masks[0]
    mask = (mask * 255).astype(np.uint8)
    return mask

def generate_masks(annotations, predictor):
    for annotation in tqdm(annotations, desc="Generating masks"):
        for bbox in annotation.bounding_boxes:
            cls = bbox[0]
            mask = generate_mask(bbox, annotation.image, predictor)
            annotation.masks.append((cls, mask))
    return annotations
            
            

def get_annotations(dataset_path, set="train"):
    train_set_path = os.path.join(dataset_path, set)
    images_path = os.path.join(train_set_path, "images")
    labels_path = os.path.join(train_set_path, "labels")
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        return None
    annotations = []
    for image_file in tqdm(os.listdir(images_path), desc="Loading annotations"):
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path)
        label_file = os.path.join(labels_path, image_file.replace(".jpg", ".txt"))
        with open(label_file, "r") as f:
            lines = f.readlines()
            bounding_boxes = []
            for line in lines: # class x_center y_center width height
                line = line.strip()
                line = line.split(" ")
                line = [float(x) for x in line]
                bounding_boxes.append(line)
            annotations.append(Annotations(image_file, bounding_boxes, image))
    return annotations

        
def get_obb(annotations, use_max_contour=False):
    for annotation in tqdm(annotations, desc="Generating OBBs"):
        for mask in annotation.masks:
            cls, mask = mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Find contours
            if len(contours) == 0:
                print(f"No contours found for {annotation.image_file}")
                continue

            if use_max_contour:
                # Get the biggest contour
                contour = max(contours, key=cv2.contourArea)
                # Get the minimum area bounding box
                rect = cv2.minAreaRect(contour)
            else:
                # Combine all contour points into a single array
                all_points = np.vstack(contours)
                # Compute the minimum area bounding rectangle for all points
                rect = cv2.minAreaRect(all_points)

            # Get the four corners of the rectangle
            box = cv2.boxPoints(rect)
            # normalize the box
            x1, y1, x2, y2, x3, y3, x4, y4 = box.flatten()
            x1, y1, x2, y2, x3, y3, x4, y4 = utils.normalize_obb_coords(x1, y1, x2, y2, x3, y3, x4, y4, annotation.image.shape[1], annotation.image.shape[0])
            # add cls to the box
            obb = [cls, x1, y1, x2, y2, x3, y3, x4, y4]
            annotation.obb.append(obb)
        
    return annotations
        

def create_yolo_obb_dataset(annotations, output_path, set="train"):
    images_path = os.path.join(output_path, set, "images")
    labels_path = os.path.join(output_path, set, "labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    for annotation in tqdm(annotations, desc="Saving YOLO OBB dataset"):
        file_name = os.path.basename(annotation.image_file)
        image_path = os.path.join(images_path, file_name)
        cv2.imwrite(image_path, annotation.image)
        label_path = os.path.join(labels_path, file_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for i, box in enumerate(annotation.obb):
                f.write(f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]} {box[7]} {box[8]}\n")


                

def main(args):
    dataset_path = args.dataset_path
    output_path = args.output_path
    device = args.device
    use_max_contour = args.use_max_contour
    device = configure_device(device)
    if os.path.exists(output_path):
        # remove the output path
        shutil.rmtree(output_path)
    for set in ["train", "valid", "test"]:
        annotations = get_annotations(dataset_path, set)
        if annotations is None:
            print(f"No annotations found for {set} set")
            continue
        print(f"Generating OBB labels for {set} set")
        predictor = initialize_predictor(args.model_cfg, args.checkpoint_path, device)
        print("Predictor initialized")
        annotations = generate_masks(annotations, predictor) 
        print("Masks generated")
        annotations = get_obb(annotations, use_max_contour)
        print("OBB labels generated")
        create_yolo_obb_dataset(annotations, output_path, set)
        print(f"OBB labels created for {set} set")
    print("OBB labels generation completed, output saved to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OBB labels from BB labels using SAM2.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset containing train/val/test sets.")
    parser.add_argument("output_path", type=str, help="Path to save the generated OBB dataset.")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to the SAM2 model configuration file.")
    path = Path.home() / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    parser.add_argument("--checkpoint_path", type=str, default=path, help="Path to the SAM2 model checkpoint.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on.")
    parser.add_argument("--use_max_contour", type=bool, default=False, help="Whether to use the max contour to generate OBBs. If not, uses the whole mask.")
    args = parser.parse_args()
    main(args)
