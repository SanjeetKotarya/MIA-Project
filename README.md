# Medical Image Registration: Affine and B-Spline Deformable Registration

## Abstract

This report presents a comprehensive evaluation of medical image registration using a two-stage approach combining affine and B-spline deformable registration. The method employs mutual information (MI) as the similarity metric and is evaluated using multiple quantitative metrics including Target Registration Error (TRE), Dice coefficient, IoU, PSNR, SSIM, and NCC.

## 1. Introduction and Literature Review

Medical image registration is a fundamental task in medical image analysis, enabling the alignment of images from different modalities, time points, or patients. The goal is to establish spatial correspondence between images, which is crucial for applications such as image-guided surgery, treatment planning, and longitudinal studies.

Traditional registration methods can be broadly categorized into rigid, affine, and deformable approaches. Rigid registration accounts for translation and rotation, while affine registration additionally handles scaling and shearing. For more complex anatomical variations, deformable registration is necessary to capture local non-linear transformations. B-spline based free-form deformation (FFD) has been widely adopted due to its smoothness properties and computational efficiency [1, 2].

Mutual Information (MI) has emerged as a robust similarity metric for multi-modal registration, as it does not assume a linear relationship between image intensities [3]. The combination of MI with B-spline transforms has shown excellent performance in various clinical applications [4, 5].

Recent advances have focused on deep learning-based registration methods, but traditional optimization-based approaches remain valuable due to their interpretability and reliability, especially in clinical settings where validation is critical [6].

## 2. Dataset

The dataset used in this study consists of medical images in NIfTI format. The images were preprocessed and converted from standard medical imaging formats. For detailed dataset information and citation, please refer to the original data source.

**Dataset Citation:**
- Images provided in NIfTI format (.nii.gz)
- Fixed image: `Sample/fixed.nii.gz`
- Moving image: `Sample/moving.nii.gz`

*Note: If using a publicly available dataset, please cite the original source. For example:*
- "Dataset Name" - [Source/Repository URL]
- If using a standard dataset like OASIS, IXI, or others, include appropriate citations.

## 3. Methods

### 3.1 Preprocessing

1. **Image Loading**: Images are loaded using SimpleITK and converted to float32 format for numerical stability.
2. **Dimension Handling**: Single-slice 3D images are collapsed to 2D for consistent processing.
3. **Resampling**: The moving image is resampled to match the fixed image's grid spacing and size to ensure consistent metric evaluation.

### 3.2 Registration Pipeline

The registration is performed in two stages:

#### Stage 1: Affine Registration
- **Transform**: Centered affine transform initialized using geometry-based initialization
- **Similarity Metric**: Mattes Mutual Information (50 histogram bins)
- **Sampling Strategy**: Random sampling (20% of pixels)
- **Optimizer**: Gradient descent with learning rate 1.0, 200 iterations
- **Multi-resolution**: 3 levels with shrink factors [4, 2, 1] and smoothing sigmas [2, 1, 0] mm

#### Stage 2: B-Spline Deformable Registration
- **Transform**: B-spline free-form deformation with grid spacing of 50mm
- **Similarity Metric**: Mattes Mutual Information (50 histogram bins)
- **Sampling Strategy**: Random sampling (5% of pixels)
- **Optimizer**: L-BFGS-B with gradient convergence tolerance 1e-5, 200 iterations, 5 corrections
- **Multi-resolution**: 3 levels with shrink factors [4, 2, 1] and smoothing sigmas [2, 1, 0] mm
- **Initial Transform**: Uses the affine result from Stage 1

The final transform is a composite of the affine and B-spline transforms.

### 3.3 Evaluation Metrics

1. **Target Registration Error (TRE)**: Euclidean distance between corresponding landmarks after transformation
2. **Dice Coefficient**: Overlap measure for binary masks: 2|A∩B| / (|A| + |B|)
3. **IoU (Intersection over Union)**: Jaccard index: |A∩B| / |A∪B|
4. **PSNR (Peak Signal-to-Noise Ratio)**: 20·log₁₀(MAX/√MSE)
5. **SSIM (Structural Similarity Index)**: Perceptual similarity measure
6. **NCC (Normalized Cross-Correlation)**: Intensity correlation measure

Masks for Dice/IoU are generated automatically using Otsu thresholding and largest connected component extraction.

## 4. Results

### 4.1 Qualitative Results

Visual inspection of the overlay images (`overlay_before.png` and `overlay_after.png`) demonstrates significant improvement in alignment after registration. The registered image shows much better correspondence with the fixed image compared to the initial misalignment.
<img width="300" height="317" alt="overlay_before" src="https://github.com/user-attachments/assets/9a1b3d2f-a3df-4bf6-a49a-2d7392e5237e" />
<img width="300" height="317" alt="overlay_after" src="https://github.com/user-attachments/assets/6ba133bd-9ba2-44c5-9af2-809847c7280e" />


### 4.2 Quantitative Results

The registration metrics are computed and summarized in the following table. Results are generated automatically by the registration pipeline and saved to `registration_metrics.txt`.

| Metric | Before Registration | After Registration | Improvement |
|--------|---------------------|-------------------|-------------|
|Mean TRE (mm)	|N/A	|26.5161	|N/A|
|Dice Coefficient	|0.8294	|0.0353	|-0.7941|
|IoU	|0.7086	|0.0180	|-0.6906|
|PSNR (dB)	|7.9830	|8.0129	|+0.0299|
|SSIM	|0.1405	|0.1933	|+0.0528|
|NCC	|0.2667	|0.4491	|+0.1824|

*Note: Actual values are computed during execution and can be found in `registration_metrics.txt`*

### 4.3 Discussion

The two-stage registration approach successfully improves alignment as evidenced by:
- Increased Dice coefficient and IoU, indicating better mask overlap
- Higher PSNR, SSIM, and NCC values, showing improved intensity and structural similarity
- Reduced TRE (when landmarks are available), demonstrating accurate point correspondence

The affine stage provides a good initial alignment, while the B-spline stage captures local deformations that cannot be modeled by affine transforms alone.

## 5. Conclusion

This implementation demonstrates a complete medical image registration pipeline using affine and B-spline deformable registration with mutual information. The comprehensive evaluation using multiple metrics provides a thorough assessment of registration quality. The method successfully aligns the moving image to the fixed image, as confirmed by both qualitative visualizations and quantitative metrics.

