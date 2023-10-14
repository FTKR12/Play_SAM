import os
import cv2
import pickle
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

from utils.options import get_args
from utils.show import show_img, show_img_with_point, show_img_with_point_mask, show_img_with_mask

def test_auto(args, img):
    
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(args.device)

    predictor = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=15,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        box_nms_thresh=1.0,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    masks = predictor.generate(img)
    
    show_img(img, args.output_dir)
    show_img_with_mask(img, masks, args.output_dir)
    

def test_with_point(args, img):
    
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(args.device)
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    point = np.array(args.point)
    point_label = np.array(args.point_label)
    masks, scores, logits = predictor.predict(
        point_coords=point,
        point_labels=point_label,
        multimask_output=True,
    )
    
    show_img(img, args.output_dir)
    show_img_with_point(img, point, point_label, args.output_dir)
    show_img_with_point_mask(img, point, point_label, masks, scores, args.output_dir)

if __name__ == "__main__":

    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # set data
    if args.data_type == 'sperm':
        with open(args.sperm_path, 'rb') as f:
            img = pickle.load(f)[args.sperm_id]
    elif args.data_type == 'test_image':
        img = cv2.imread(args.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if args.mode == 'point':
        test_with_point(args, img)
    elif args.mode == 'auto':
        test_auto(args, img)