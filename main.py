import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Try to import optional dependencies
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    print("Warning: scikit-image not found. SSIM will be skipped.")
    HAS_SSIM = False

try:
    from scipy.ndimage import label
    HAS_SCIPY = True
except ImportError:
    print("Warning: scipy not found. Using SimpleITK for connected components.")
    HAS_SCIPY = False

# ---------- Helper functions ----------
def read_image(path):
    return sitk.ReadImage(path, sitk.sitkFloat32)

def resample_to_reference(moving, reference):
    # Resample moving image to reference grid (linear)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving)

def show_overlay_2d(fixed_np, moving_np, out_path=None):
    # simple overlay plot (alpha blend)
    plt.figure(figsize=(8,8))
    plt.imshow(fixed_np, cmap='gray')
    plt.imshow(moving_np, cmap='jet', alpha=0.4)
    plt.axis('off')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

def itk_to_numpy(img):
    arr = sitk.GetArrayFromImage(img) # z,y,x
    # for 2D images, array shape is (1,y,x) or (y,x)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr

def transform_points(transform, points):
    # points: Nx3 or Nx2 in physical coords
    out = []
    for p in points:
        out.append(transform.TransformPoint(tuple(p)))
    return np.array(out)

def squeeze_if_single_slice(img):
    if img.GetDimension() == 3 and img.GetSize()[-1] == 1:
        size = list(img.GetSize())
        size[-1] = 0  # collapse last dimension
        index = [0] * img.GetDimension()
        return sitk.Extract(img, size, index)
    return img

def _landmarks_dim_matches(points: np.ndarray, dim: int) -> bool:
    if points is None:
        return False
    if not isinstance(points, np.ndarray):
        return False
    if points.ndim != 2:
        return False
    return points.shape[1] == dim

def create_brain_mask(img_sitk, threshold_percentile=10):
    """Create a brain mask using Otsu thresholding and largest connected component."""
    # Otsu thresholding
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    mask_sitk = otsu_filter.Execute(img_sitk)
    
    # Get largest connected component
    if HAS_SCIPY:
        # Use scipy for connected components
        mask_np = sitk.GetArrayFromImage(mask_sitk)
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        labeled, num_features = label(mask_np)
        if num_features > 0:
            largest_component = labeled == np.argmax(np.bincount(labeled.flat)[1:]) + 1
            mask_np = largest_component.astype(np.uint8)
        else:
            mask_np = (mask_np > 0).astype(np.uint8)
        return mask_np.astype(bool)
    else:
        # Use SimpleITK for connected components (fallback)
        cc_filter = sitk.ConnectedComponentImageFilter()
        labeled_sitk = cc_filter.Execute(mask_sitk)
        labeled_np = sitk.GetArrayFromImage(labeled_sitk)
        if labeled_np.ndim == 3 and labeled_np.shape[0] == 1:
            labeled_np = labeled_np[0]
        
        # Get largest component
        unique_labels, counts = np.unique(labeled_np, return_counts=True)
        if len(unique_labels) > 1:  # More than just background (0)
            # Find largest non-zero component
            non_zero_mask = unique_labels != 0
            non_zero_labels = unique_labels[non_zero_mask]
            non_zero_counts = counts[non_zero_mask]
            if len(non_zero_labels) > 0:
                largest_idx = np.argmax(non_zero_counts)
                largest_label = non_zero_labels[largest_idx]
                mask_np = (labeled_np == largest_label).astype(np.uint8)
            else:
                mask_np = (labeled_np > 0).astype(np.uint8)
        else:
            mask_np = (labeled_np > 0).astype(np.uint8)
        
        return mask_np.astype(bool)

def compute_dice_iou(mask1, mask2):
    """Compute Dice coefficient and IoU (Jaccard index) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    dice = (2.0 * intersection) / (mask1.sum() + mask2.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    return dice, iou

def normalize_image(img_np):
    """Normalize image to [0, 1] range."""
    img_min = img_np.min()
    img_max = img_np.max()
    if img_max > img_min:
        return (img_np - img_min) / (img_max - img_min)
    return img_np

def compute_psnr(img1, img2, data_range=None):
    """Compute PSNR between two images."""
    img1 = normalize_image(img1)
    img2 = normalize_image(img2)
    if data_range is None:
        data_range = 1.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))

def compute_ssim_metric(img1, img2):
    """Compute SSIM between two images."""
    if not HAS_SSIM:
        return None
    img1 = normalize_image(img1)
    img2 = normalize_image(img2)
    # SSIM works on 2D images, so handle 3D by taking middle slice or flattening
    if img1.ndim == 3:
        mid = img1.shape[0] // 2
        img1 = img1[mid]
        img2 = img2[mid]
    return ssim(img1, img2, data_range=1.0)

def compute_ncc(img1, img2):
    """Compute Normalized Cross-Correlation between two images."""
    img1 = normalize_image(img1)
    img2 = normalize_image(img2)
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    img1_mean = img1_flat.mean()
    img2_mean = img2_flat.mean()
    numerator = ((img1_flat - img1_mean) * (img2_flat - img2_mean)).sum()
    denom1 = np.sqrt(((img1_flat - img1_mean) ** 2).sum())
    denom2 = np.sqrt(((img2_flat - img2_mean) ** 2).sum())
    if denom1 == 0 or denom2 == 0:
        return 0.0
    return numerator / (denom1 * denom2)

def generate_landmarks_from_image(img_sitk, n_points=10):
    """Generate landmark points automatically from image features (corners, center, etc.)."""
    dim = img_sitk.GetDimension()
    size = img_sitk.GetSize()
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    
    landmarks = []
    if dim == 2:
        # Generate points: corners, center, midpoints of edges
        corners = [
            [origin[0], origin[1]],
            [origin[0] + size[0] * spacing[0], origin[1]],
            [origin[0], origin[1] + size[1] * spacing[1]],
            [origin[0] + size[0] * spacing[0], origin[1] + size[1] * spacing[1]]
        ]
        center = [
            origin[0] + size[0] * spacing[0] / 2,
            origin[1] + size[1] * spacing[1] / 2
        ]
        midpoints = [
            [origin[0] + size[0] * spacing[0] / 2, origin[1]],
            [origin[0] + size[0] * spacing[0] / 2, origin[1] + size[1] * spacing[1]],
            [origin[0], origin[1] + size[1] * spacing[1] / 2],
            [origin[0] + size[0] * spacing[0], origin[1] + size[1] * spacing[1] / 2]
        ]
        landmarks = corners + [center] + midpoints[:min(4, n_points - 5)]
    else:  # 3D
        # Similar approach for 3D
        corners_3d = []
        for x in [0, size[0]]:
            for y in [0, size[1]]:
                for z in [0, size[2]]:
                    corners_3d.append([
                        origin[0] + x * spacing[0],
                        origin[1] + y * spacing[1],
                        origin[2] + z * spacing[2]
                    ])
        center = [
            origin[0] + size[0] * spacing[0] / 2,
            origin[1] + size[1] * spacing[1] / 2,
            origin[2] + size[2] * spacing[2] / 2
        ]
        landmarks = corners_3d[:min(len(corners_3d), n_points - 1)] + [center]
    
    return np.array(landmarks[:n_points])

# ---------- Preprocessing ----------
fixed_path = "Sample/fixed.nii.gz"   # provide these
moving_path = "Sample/moving.nii.gz" # provide these
if not os.path.exists(fixed_path) or not os.path.exists(moving_path):
    print("Input images not found. Set 'fixed_path' and 'moving_path' to valid files.")
    print(f"fixed_path exists: {os.path.exists(fixed_path)} -> {fixed_path}")
    print(f"moving_path exists: {os.path.exists(moving_path)} -> {moving_path}")
    sys.exit(1)

fixed = read_image(fixed_path)
moving = read_image(moving_path)
fixed = squeeze_if_single_slice(fixed)
moving = squeeze_if_single_slice(moving)

# resample moving to fixed grid to make metric evaluations consistent
moving_rs = resample_to_reference(moving, fixed)

# Check image size - smoothing needs at least 4 pixels per dimension
fixed_size = fixed.GetSize()
min_size = min(fixed_size)
# Adjust parameters for small images - disable multi-resolution if any dimension < 4
use_multiresolution = min_size >= 4

# ---------- AFFINE REGISTRATION (MI similarity) ----------
affine_tx = sitk.CenteredTransformInitializer(
    fixed, moving_rs,
    sitk.AffineTransform(fixed.GetDimension()),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

affine_reg = sitk.ImageRegistrationMethod()
affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
affine_reg.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
affine_reg.SetMetricSamplingPercentage(0.2)
affine_reg.SetInterpolator(sitk.sitkLinear)
affine_reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
affine_reg.SetOptimizerScalesFromPhysicalShift()
affine_reg.SetInitialTransform(affine_tx, inPlace=False)
if use_multiresolution:
    affine_reg.SetShrinkFactorsPerLevel([4, 2, 1])
    affine_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    affine_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

print("Running affine registration...")
affine_result_tx = affine_reg.Execute(fixed, moving_rs)
print("Affine optimizer stop condition:", affine_reg.GetOptimizerStopConditionDescription())
print("Affine metric value:", affine_reg.GetMetricValue())

# Resample moving using affine result
moving_affine_resampled = sitk.Resample(moving_rs, fixed, affine_result_tx, sitk.sitkLinear, 0.0, moving_rs.GetPixelID())

# ---------- B-SPLINE DEFORMABLE REGISTRATION (MI) ----------
# Setup B-spline grid (coarse to fine: adjust grid spacing for complexity)
grid_physical_spacing = [50.0, 50.0, 50.0] if fixed.GetDimension()==3 else [50.0, 50.0]
image_physical_size = [sz*sp for sz, sp in zip(fixed.GetSize(), fixed.GetSpacing())]
mesh_size = [int(np.round(image_physical_size[i]/grid_physical_spacing[i])) for i in range(len(grid_physical_spacing))]
# ensure mesh size >= 1
mesh_size = [max(1, m-1) for m in mesh_size]  # ANTs-like

transform_domain_mesh_size = mesh_size
bspline_transform = sitk.BSplineTransformInitializer(image1=fixed, transformDomainMeshSize=transform_domain_mesh_size, order=3)

bspline_reg = sitk.ImageRegistrationMethod()
bspline_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
bspline_reg.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
bspline_reg.SetMetricSamplingPercentage(0.05)
bspline_reg.SetInterpolator(sitk.sitkLinear)
bspline_reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=200, maximumNumberOfCorrections=5)
bspline_reg.SetInitialTransform(bspline_transform, inPlace=False)
# Use the affine result as the moving initial transform; optimize the BSpline
bspline_reg.SetMovingInitialTransform(affine_result_tx)

if use_multiresolution:
    bspline_reg.SetShrinkFactorsPerLevel([4, 2, 1])
    bspline_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

print("Running B-spline deformable registration...")
final_transform = bspline_reg.Execute(fixed, moving_rs)
print("BSpline optimizer stop condition:", bspline_reg.GetOptimizerStopConditionDescription())
print("BSpline metric value:", bspline_reg.GetMetricValue())

# Compose transforms: affine then bspline
full_transform = sitk.CompositeTransform(fixed.GetDimension())
full_transform.AddTransform(affine_result_tx)
full_transform.AddTransform(final_transform)

# Resample final moving image
moving_final = sitk.Resample(moving_rs, fixed, full_transform, sitk.sitkLinear, 0.0, moving_rs.GetPixelID())

# ---------- METRICS CALCULATION ----------
print("\n" + "="*60)
print("COMPUTING REGISTRATION METRICS")
print("="*60)

# Convert to numpy for metric calculations
fixed_np = itk_to_numpy(fixed)
moving_before_np = itk_to_numpy(moving_rs)
moving_after_np = itk_to_numpy(moving_final)

# Normalize for intensity-based metrics
fixed_norm = normalize_image(fixed_np)
moving_before_norm = normalize_image(moving_before_np)
moving_after_norm = normalize_image(moving_after_np)

# 1. TRE (Target Registration Error)
print("\n--- TRE (Target Registration Error) ---")
dim = fixed.GetDimension()
# Try to generate landmarks automatically, or use manual ones if provided
fixed_landmarks = None
moving_landmarks = None

# Option: Generate automatic landmarks (geometric points)
# For real evaluation, you should provide manually annotated landmarks
USE_AUTO_LANDMARKS = True  # Set to False to use manual landmarks
if USE_AUTO_LANDMARKS:
    try:
        fixed_landmarks = generate_landmarks_from_image(fixed, n_points=9)
        # Generate corresponding points in moving image (before registration)
        moving_landmarks = generate_landmarks_from_image(moving_rs, n_points=9)
        print(f"Generated {len(fixed_landmarks)} automatic landmark pairs")
    except Exception as e:
        print(f"Auto-landmark generation failed: {e}")
        fixed_landmarks = None
        moving_landmarks = None

if _landmarks_dim_matches(fixed_landmarks, dim) and _landmarks_dim_matches(moving_landmarks, dim):
    moving_landmarks_transformed = transform_points(full_transform, moving_landmarks)
    dists = np.linalg.norm(moving_landmarks_transformed - fixed_landmarks, axis=1)
    mean_tre = dists.mean()
    std_tre = dists.std()
    print(f"TRE per landmark (mm): {dists}")
    print(f"Mean TRE: {mean_tre:.4f} mm, Std: {std_tre:.4f} mm")
else:
    mean_tre = None
    std_tre = None
    print("TRE: Not computed (no landmarks provided)")

# 2. Dice Coefficient and IoU (using automatic brain masks)
print("\n--- Dice Coefficient & IoU ---")
try:
    fixed_mask = create_brain_mask(fixed)
    moving_before_mask = create_brain_mask(moving_rs)
    moving_after_mask = create_brain_mask(moving_final)
    
    # Dice/IoU before registration
    dice_before, iou_before = compute_dice_iou(fixed_mask, moving_before_mask)
    print(f"Before registration - Dice: {dice_before:.4f}, IoU: {iou_before:.4f}")
    
    # Dice/IoU after registration
    dice_after, iou_after = compute_dice_iou(fixed_mask, moving_after_mask)
    print(f"After registration  - Dice: {dice_after:.4f}, IoU: {iou_after:.4f}")
    print(f"Improvement - Dice: {dice_after - dice_before:+.4f}, IoU: {iou_after - iou_before:+.4f}")
except Exception as e:
    print(f"Dice/IoU computation failed: {e}")
    dice_before = dice_after = iou_before = iou_after = None

# 3. PSNR (Peak Signal-to-Noise Ratio)
print("\n--- PSNR (Peak Signal-to-Noise Ratio) ---")
psnr_before = compute_psnr(fixed_norm, moving_before_norm)
psnr_after = compute_psnr(fixed_norm, moving_after_norm)
print(f"Before registration: {psnr_before:.4f} dB")
print(f"After registration:  {psnr_after:.4f} dB")
print(f"Improvement: {psnr_after - psnr_before:+.4f} dB")

# 4. SSIM (Structural Similarity Index)
print("\n--- SSIM (Structural Similarity Index) ---")
try:
    ssim_before = compute_ssim_metric(fixed_norm, moving_before_norm)
    ssim_after = compute_ssim_metric(fixed_norm, moving_after_norm)
    print(f"Before registration: {ssim_before:.4f}")
    print(f"After registration:  {ssim_after:.4f}")
    print(f"Improvement: {ssim_after - ssim_before:+.4f}")
except Exception as e:
    print(f"SSIM computation failed: {e}")
    ssim_before = ssim_after = None

# 5. NCC (Normalized Cross-Correlation)
print("\n--- NCC (Normalized Cross-Correlation) ---")
ncc_before = compute_ncc(fixed_norm, moving_before_norm)
ncc_after = compute_ncc(fixed_norm, moving_after_norm)
print(f"Before registration: {ncc_before:.4f}")
print(f"After registration:  {ncc_after:.4f}")
print(f"Improvement: {ncc_after - ncc_before:+.4f}")

# Summary table
print("\n" + "="*60)
print("METRICS SUMMARY TABLE")
print("="*60)
print(f"{'Metric':<20} {'Before':<15} {'After':<15} {'Improvement':<15}")
print("-"*60)
if mean_tre is not None:
    print(f"{'Mean TRE (mm)':<20} {'N/A':<15} {mean_tre:<15.4f} {'N/A':<15}")
if dice_before is not None:
    print(f"{'Dice Coefficient':<20} {dice_before:<15.4f} {dice_after:<15.4f} {dice_after-dice_before:<15.4f}")
    print(f"{'IoU':<20} {iou_before:<15.4f} {iou_after:<15.4f} {iou_after-iou_before:<15.4f}")
print(f"{'PSNR (dB)':<20} {psnr_before:<15.4f} {psnr_after:<15.4f} {psnr_after-psnr_before:<15.4f}")
if ssim_before is not None:
    print(f"{'SSIM':<20} {ssim_before:<15.4f} {ssim_after:<15.4f} {ssim_after-ssim_before:<15.4f}")
print(f"{'NCC':<20} {ncc_before:<15.4f} {ncc_after:<15.4f} {ncc_after-ncc_before:<15.4f}")
print("="*60)

# ---------- Save visualizations ----------
# (fixed_np, moving_before_np, moving_after_np already computed above)

# If 3D, pick middle slice for visualization
if fixed_np.ndim == 3:
    mid = fixed_np.shape[0] // 2
    fixed_vis = fixed_np[mid]
    moving_before_vis = moving_before_np[mid]
    moving_after_vis = moving_after_np[mid]
else:
    fixed_vis = fixed_np
    moving_before_vis = moving_before_np
    moving_after_vis = moving_after_np

show_overlay_2d(fixed_vis, moving_before_vis, out_path="overlay_before.png")
show_overlay_2d(fixed_vis, moving_after_vis, out_path="overlay_after.png")

print("\nSaved overlay_before.png and overlay_after.png")

# ---------- Save metrics to file ----------
def save_metrics_report(metrics_dict, filename="registration_metrics.txt"):
    """Save metrics to a text file."""
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("REGISTRATION METRICS REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Metric':<25} {'Before':<15} {'After':<15} {'Improvement':<15}\n")
        f.write("-"*60 + "\n")
        for metric, values in metrics_dict.items():
            before_str = f"{values['before']:.4f}" if values['before'] is not None else "N/A"
            after_str = f"{values['after']:.4f}" if values['after'] is not None else "N/A"
            improvement_str = f"{values['improvement']:.4f}" if values['improvement'] is not None else "N/A"
            f.write(f"{metric:<25} {before_str:<15} {after_str:<15} {improvement_str:<15}\n")
        f.write("="*60 + "\n")
    print(f"\nMetrics saved to {filename}")

# Prepare metrics dictionary
metrics_dict = {}
if mean_tre is not None:
    metrics_dict['Mean TRE (mm)'] = {'before': None, 'after': mean_tre, 'improvement': None}
if dice_before is not None:
    metrics_dict['Dice Coefficient'] = {'before': dice_before, 'after': dice_after, 'improvement': dice_after - dice_before}
    metrics_dict['IoU'] = {'before': iou_before, 'after': iou_after, 'improvement': iou_after - iou_before}
metrics_dict['PSNR (dB)'] = {'before': psnr_before, 'after': psnr_after, 'improvement': psnr_after - psnr_before}
if ssim_before is not None:
    metrics_dict['SSIM'] = {'before': ssim_before, 'after': ssim_after, 'improvement': ssim_after - ssim_before}
metrics_dict['NCC'] = {'before': ncc_before, 'after': ncc_after, 'improvement': ncc_after - ncc_before}

save_metrics_report(metrics_dict)
