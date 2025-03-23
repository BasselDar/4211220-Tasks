# Video Captioning using CNN-LSTM

This project implements a video captioning system using a CNN-LSTM architecture. The model uses a CNN to extract features from video frames and an LSTM to generate captions for the video.

## Project Structure

- `data/`: Contains the dataset with images and captions
- `model.py`: Contains the CNN-LSTM model architecture
- `train.py`: Script to train the model
- `predict.py`: Script to generate captions for new videos
- `utils.py`: Utility functions for data processing
- `video_to_frames.py`: Script to extract frames from videos

## Requirements

Install the required packages using:

```
pip install -r requirements.txt
```

## Usage

1. Train the model:
   ```
   python train.py
   ```

2. Generate captions for a video:
   ```
   python predict.py --video path/to/video.mp4
   ```

## Model Architecture

The model uses:
- CNN: Pre-trained InceptionV3 for feature extraction from video frames
- LSTM: For sequence generation (captions)
- Word Embeddings: To represent words in the captions

The model is trained on a dataset of images with corresponding captions, then applied to videos by extracting frames and predicting captions for the video as a whole. 