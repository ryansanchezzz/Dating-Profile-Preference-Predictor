___Dating Profile Preference Predictor___
Note: This project was originally developed in a Colab notebook (.ipynb) and the code has been reorganized for readability and clarity.

A deep learning project that predicts user interest in dating profiles based on previous swipe and preference data. The model uses multiple profile images per user and leverages computer vision to identify patterns that correlate with user preferences.

___Project Overview___
* Developed a CNN-based model using a pretrained ResNet18 backbone to predict “like” or “dislike” for dating profiles.
* Built a custom PyTorch dataset to handle multiple images per profile, with support for batch processing and data augmentation.
* Evaluated the model on unseen profiles, achieving ~80% classification accuracy.
* Incorporated Grad-CAM visualizations to interpret what the model focuses on in profile images.

___Features___
* Handles profiles with multiple images (6 images per profile by default).
* Applies data augmentation for training: random flips, rotations, color jitter, and normalization.
* Implements early stopping and model checkpointing during training.
* Outputs confusion matrix, classification report, and Grad-CAM heatmaps for model interpretability.

___Dataset___
* Input: Each profile contains a folder with 6+ images and a label file (label.txt or rating.txt).
* Labels: like/1 or dislike/0 mapped to binary classification.

Train/Test split: 80% / 20% by default.

___Dependencies___
pip install -r requirements.txt
