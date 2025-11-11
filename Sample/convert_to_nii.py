# convert_to_nii.py
import SimpleITK as sitk
import numpy as np
import cv2
import os

def jpg_to_clean_nii(in_jpg, out_nii, spacing=(1.0, 1.0)):
    # resolve paths relative to this script's directory
    base_dir = os.path.dirname(__file__)
    in_path = in_jpg if os.path.isabs(in_jpg) else os.path.join(base_dir, in_jpg)
    out_path = out_nii if os.path.isabs(out_nii) else os.path.join(base_dir, out_nii)

    # read grayscale
    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Input not found or unreadable: {in_path}")
        return
    # tight crop around head (removes scale bar/borders)
    thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        pad = 5
        x = max(0, x+pad); y = max(0, y+pad)
        w = max(1, w-2*pad); h = max(1, h-2*pad)
        img = img[y:y+h, x:x+w]

    # make float, z-dimension=1 for 2D NIfTI
    arr = np.expand_dims(img.astype(np.float32), axis=0)
    nii = sitk.GetImageFromArray(arr)     # (z,y,x)
    if nii.GetDimension() == 3:
        nii.SetSpacing(spacing + (1.0,))
    else:
        nii.SetSpacing(spacing)
    sitk.WriteImage(nii, out_path)
    print("Saved:", out_path)

jpg_to_clean_nii("ct.png",  "fixed.nii.gz")
jpg_to_clean_nii("mri.png", "moving.nii.gz")
