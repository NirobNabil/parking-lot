from ultralytics import YOLO, RTDETR

model = RTDETR("best.pt")

dataset_base_dir = "F:/thesis/datasets/quattro"

results = model([dataset_base_dir+"IMG20241003212249.jpg", dataset_base_dir+"IMG20241003212307.jpg"])

for i, result in enumerate(results):
    # boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()
    # result.save(filename=f"result_{i}.jpg")  # save to disk