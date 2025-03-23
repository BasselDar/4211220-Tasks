import os
import argparse
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from utils import create_feature_extractor, extract_features, word_for_id
from video_to_frames import extract_frames

def generate_caption(model, tokenizer, image_features, max_length):
    """Generate a caption for the given image features"""
    # Seed with 'start' token
    in_text = 'startseq'
    
    # Initialize result
    result = []
    
    # Iterate until max length or end token
    for i in range(max_length):
        # Encode the current input text
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict the next word
        yhat = model.predict([image_features, sequence], verbose=0)
        # Get the index of the word with highest probability
        yhat = np.argmax(yhat)
        # Convert the index to a word
        word = word_for_id(yhat, tokenizer)
        
        # Stop if we can't find the word
        if word is None:
            break
            
        # Add the word to the result if it's not a special token
        if word not in ['startseq', 'endseq']:
            result.append(word)
            
        # Add the word to the input text    
        in_text += ' ' + word
        
        # Stop if we predict the end token
        if word == 'endseq':
            break
            
    # Join the words to form the caption
    return ' '.join(result)

def predict_video_caption(video_path, model_path, tokenizer_path, max_length_path, temp_dir='temp_frames'):
    """Generate caption for a video"""
    # Load the model and tokenizer
    print("Loading model and tokenizer...")
    model = load_model(model_path)
    
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open(max_length_path, 'rb') as f:
        max_length = pickle.load(f)
    
    # Create the feature extractor model
    print("Creating feature extractor...")
    feature_extractor = create_feature_extractor()
    
    # Extract frames from the video
    print(f"Extracting frames from {video_path}...")
    frame_paths = extract_frames(video_path, temp_dir)
    
    if not frame_paths:
        print("No frames could be extracted from the video")
        return
    
    # Extract features from frames
    print("Extracting features from frames...")
    frame_features = []
    for frame_path in frame_paths:
        feature = extract_features(frame_path, feature_extractor)
        if feature is not None:
            frame_features.append(feature[0])  # Remove batch dimension
    
    if not frame_features:
        print("Could not extract features from any frame")
        return
    
    # Average features across frames
    print("Generating caption...")
    video_features = np.mean(frame_features, axis=0)
    video_features = np.expand_dims(video_features, axis=0)  # Add batch dimension
    
    # Generate caption
    caption = generate_caption(model, tokenizer, video_features, max_length)
    
    # Display the results
    print("\nGenerated Caption:")
    print(caption)
    
    # Optionally, display a few sample frames with the caption
    plot_frames_with_caption(frame_paths, caption)
    
    return caption

def plot_frames_with_caption(frame_paths, caption, num_frames=4):
    """Plot a few frames with the generated caption"""
    # Select a subset of frames to display
    if len(frame_paths) > num_frames:
        indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
        selected_frames = [frame_paths[i] for i in indices]
    else:
        selected_frames = frame_paths
    
    # Create a figure
    plt.figure(figsize=(15, 10))
    
    # Plot each frame
    for i, frame_path in enumerate(selected_frames):
        plt.subplot(2, 2, i + 1)
        img = Image.open(frame_path)
        plt.imshow(img)
        plt.axis('off')
    
    # Set the caption as the figure title
    plt.suptitle(f"Caption: {caption}", fontsize=16)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('caption_result.png')
    plt.close()
    print("Result saved as 'caption_result.png'")

def main():
    parser = argparse.ArgumentParser(description="Generate captions for videos")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--model", type=str, default="models/final_model.h5", help="Path to the trained model")
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer.pkl", help="Path to the tokenizer")
    parser.add_argument("--max_length", type=str, default="models/max_length.pkl", help="Path to the max length")
    parser.add_argument("--temp_dir", type=str, default="temp_frames", help="Temporary directory for frames")
    
    args = parser.parse_args()
    
    predict_video_caption(
        args.video, 
        args.model, 
        args.tokenizer, 
        args.max_length, 
        args.temp_dir
    )

if __name__ == "__main__":
    main() 