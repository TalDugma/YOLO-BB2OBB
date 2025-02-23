# YOLO-BB2OBB

Convert a **YOLO Object Detection Dataset** with standard **Bounding Box (BB)** annotations to a **YOLO Oriented Bounding Box (OBB)** dataset automatically using **Meta's Segment Anything Model 2 (SAM2)**.

## Installation

1. Follow [Meta's official guide](https://github.com/facebookresearch/sam2) to set up SAM2 as a package.
2. `cd` to the repo directory:
   ```bash
   cd YOLO-BB2OBB

3. Install the required dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To convert your dataset, run the following command:

```bash
python obb.py --dataset_path <path-to-bb-dataset> --output_path <path-to-output-obb-dataset>
```

### Example

```bash
python obb.py ./datasets/yolo_bb ./datasets/yolo_obb
```

## Description

This script utilizes **Meta's SAM2** to refine standard bounding boxes into oriented bounding boxes, improving object detection accuracy in YOLO models that support OBB annotations.

## Requirements

- Python 3.x
- Meta's SAM2
- Dependencies from `requirements.txt`

