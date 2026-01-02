# Skin Texture CNN (TensorFlow)

This project trains a CNN to predict skin texture smoothness/roughness from patch images.
Only `Patch Dataset/Skin Texture` is used.

## Labeling rubric (manual, best accuracy)
Use a 1-5 scale:
- 1 = very smooth
- 2 = smooth
- 3 = medium
- 4 = rough
- 5 = very rough

## Quick start
1) Label images:
   python labeler.py --data_dir "Patch Dataset\Skin Texture" --labels labels.csv
   Or auto-label with Laplacian variance:
   python labeler.py --data_dir "Patch Dataset\Skin Texture" --labels labels.csv --auto_labels

2) Train:
   python train.py --data_dir "Patch Dataset\Skin Texture" --labels labels.csv

3) Export to TFLite:
   python export_tflite.py --saved_model_dir model\skin_texture_saved_model

## Cheek cropping (full-face input)
If you have full-face photos, you can crop left/right cheeks using MediaPipe landmarks:
   python infer.py --data_dir "test-data-cy" --model_dir model\skin_texture_saved_model --use_cheeks

This runs landmark detection, crops left/right cheek patches, and averages their scores.
If your MediaPipe install does not expose `mp.solutions`, download a face landmarker model and run:
   python infer.py --data_dir "test-data-cy" --model_dir model\skin_texture_saved_model --use_cheeks --landmark_model models\face_landmarker.task

## Notes
- The model is a MobileNetV2 regressor with light augmentation.
- Labels are normalized to 0-1 during training for stable optimization.
- Labeling pre-sorts by Laplacian variance (rougher first). Disable with `--presort none`.
- Auto-labeling uses Laplacian variance quantiles to assign 1-5 bins. Adjust with `--auto_bins`.
