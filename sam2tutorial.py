import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image

checkponit = 'sam2_hiera_small.pt'
model_cfg = 'sam2_hiera_s.yaml'


predictor = SAM2ImagePredictor(build_sam2(model_cfg,checkponit))

image = np.array(Image.open("German_shepherd.jpeg"))
predictor.set_image(image)


input_points = np.array([[100,90],[200,200]])
input_labels = np.array([1,0])

# 预测mask
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,
)
print(masks[0])