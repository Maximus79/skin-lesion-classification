# Skin Lesion Classification (IMA205 Challenge)

A convolutional neural network pipeline that classifies dermoscopic skin lesion images into 8 distinct diagnostic classes, built for the IMA205 (Image Processing / Machine Learning for Medical Imaging) course challenge at Télécom Paris.

**Best model accuracy: ~73%** on the held-out test set.

## Overview

Skin lesion classification is a real diagnostic task: distinguishing between 8 lesion categories from dermoscopic images directly affects treatment planning. This project explores the full pipeline from raw image to prediction, combining a fine-tuned CNN with lesion segmentation and patient metadata.

## Approach

**Model.** A pretrained ResNet is fine-tuned for 8-class classification (final FC layer replaced, trained end-to-end). Several torchvision architectures were compared (ResNet, MobileNet, EfficientNet); ResNet was chosen for accuracy, accepting the higher compute cost. Hyperparameters (learning rate `1e-5`, batch size 32, 12 epochs) were tuned empirically to balance convergence speed against overfitting.

**Class imbalance.** The 8 classes are heavily imbalanced. Class weights (inverse frequency) are applied in the cross-entropy loss during training, and the corresponding test-set prior is folded into the predicted probabilities before the final decision — which measurably improved accuracy over an unweighted baseline.

**Metadata fusion.** Patient metadata (age, sex, lesion position) is not fed directly into the CNN — image models don't interpret tabular/categorical features well. Instead, a decision tree is trained separately on the metadata and its output is combined with the CNN's softmax output (a simple ensembling/bagging approach), which improved accuracy further.

**Segmentation.** A classical image-processing pipeline (thresholding, morphological opening to remove hair artifacts, contour detection, with a center-square failsafe against corner artifacts) generates lesion masks for images missing one. In practice, training on segmented images underperformed training on the original images, most likely due to inconsistent mask quality — documented in the report as a concrete negative result and a direction for future work (e.g. SVM-based or Mask R-CNN segmentation).

## Repository contents

```
notebook/   Full training & inference pipeline (PyTorch / torchvision)
report/     Write-up: methodology, results, and analysis of what did and didn't work
data/       Metadata CSVs (age, sex, lesion position, class labels) — image files are not included
sample_results/  Example prediction output format
```

Note: the raw dermoscopic images (`Train/`, `Test/`) are not included in this repository due to size and dataset licensing; the notebook expects them in the folder structure referenced in the code.

## Stack

Python, PyTorch, torchvision (ResNet, transfer learning), OpenCV (segmentation), scikit-learn (decision tree, preprocessing), pandas.

## Author

Maxime Appert
