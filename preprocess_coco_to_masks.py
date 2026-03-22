import json, os
from pathlib import Path
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

def ensure(p): 
    Path(p).mkdir(parents=True, exist_ok=True)

def rle_to_mask(rle, height, width):
    return maskUtils.decode(rle)

# Convert COCO annotations to binary masks and save them as PNG images. Each mask corresponds to one image and contains 1s for pixels belonging to any annotated object and 0s for background. This is a common format for training segmentation models. The script reads the COCO JSON file, processes each image's annotations to create a mask, and saves both the original image and the corresponding mask in separate directories. Adjust the paths as needed for your dataset structure. 

def polygons_to_mask_for_image(coco_ann, img_info):
    h, w = img_info['height'], img_info['width']
    mask = np.zeros((h, w), dtype=np.uint8)
    anns = coco_ann.getAnnIds(imgIds=[img_info['id']], iscrowd=None)
    anns = coco_ann.loadAnns(anns)
    for a in anns:
        seg = a.get('segmentation', None)
        if seg is None:
            continue
        if isinstance(seg, list):
            # polygon list
            rles = maskUtils.frPyObjects(seg, h, w)
            rle = maskUtils.merge(rles)
            m = maskUtils.decode(rle)
        else:
            # already RLE
            m = maskUtils.decode(seg)
        mask = np.maximum(mask, (m>0).astype(np.uint8))
    return mask

 # Example usage:
def process(coco_json_path, images_root, out_images_root, out_masks_root):
    coco = COCO(str(coco_json_path))
    ensure(out_images_root)
    ensure(out_masks_root)
    for img in coco.loadImgs(coco.getImgIds()):
        src = Path(images_root) / img['file_name']
        dst_img = Path(out_images_root) / img['file_name']
        dst_mask = Path(out_masks_root) / Path(img['file_name']).with_suffix('.png').name
        if not src.exists():
            print("missing:", src); continue
        # copy image
        if not dst_img.exists():
            Path(dst_img).parent.mkdir(parents=True, exist_ok=True)
            Image.open(src).save(dst_img)
        # build mask
        mask = polygons_to_mask_for_image(coco, img)
        Image.fromarray((mask*255).astype(np.uint8)).save(dst_mask)

if __name__ == "__main__":
    # Paths (edit to where you unpacked RipVIS)
    base = Path(".")
    # train
    process(base/"train"/"coco_annotations"/"train.json",
            base/"train"/"sampled_images"/"images",
            "data/train/images",
            "data/train/masks")
    # val
    process(base/"val"/"coco_annotations"/"val.json",
            base/"val"/"sampled_images"/"images",
            "data/val/images",
            "data/val/masks")