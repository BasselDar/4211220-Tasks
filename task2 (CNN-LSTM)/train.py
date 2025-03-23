import os
import pickle
import argparse
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from utils import (
    load_captions, preprocess_captions, create_tokenizer, 
    max_caption_length, create_feature_extractor, 
    load_image_features, create_data_generator
)
from model import define_model, define_captioning_model

def train(args):
    """Train the CNN-LSTM model for image/video captioning"""
    
    # Create output directories if they don't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    print("Loading and preprocessing captions...")
    captions_dict = load_captions(args.captions_path)
    
    # Preprocess captions
    captions_dict = preprocess_captions(captions_dict)
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(captions_dict)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")
    
    # Find maximum caption length
    max_length = max_caption_length(captions_dict)
    print(f"Maximum Caption Length: {max_length}")
    
    # Save tokenizer and max_length for later use
    with open(os.path.join(args.model_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(args.model_dir, 'max_length.pkl'), 'wb') as f:
        pickle.dump(max_length, f)
    
    # Create feature extractor (CNN) model
    print("Creating feature extractor...")
    feature_extractor = create_feature_extractor()
    
    # Extract features from images
    print("Extracting features from images...")
    features = load_image_features(args.images_dir, feature_extractor)
    
    # Save features for later use
    with open(os.path.join(args.model_dir, 'features.pkl'), 'wb') as f:
        pickle.dump(features, f)
    
    # Calculate dataset sizes
    num_images = len(features)
    train_size = int(num_images * 0.8)
    val_size = num_images - train_size
    
    # Get list of image names
    image_names = list(features.keys())
    np.random.shuffle(image_names)
    
    # Split into training and validation sets
    train_images = image_names[:train_size]
    val_images = image_names[train_size:]
    
    # Create dictionaries for training and validation sets
    train_features = {img: features[img] for img in train_images if img in features}
    val_features = {img: features[img] for img in val_images if img in features}
    
    train_captions = {img: captions_dict[img] for img in train_images if img in captions_dict}
    val_captions = {img: captions_dict[img] for img in val_images if img in captions_dict}
    
    # Create data generators
    print("Creating data generators...")
    train_generator = create_data_generator(
        tokenizer, max_length, train_captions, train_features, 
        vocab_size, batch_size=args.batch_size
    )
    val_generator = create_data_generator(
        tokenizer, max_length, val_captions, val_features, 
        vocab_size, batch_size=args.batch_size
    )
    
    # Determine steps per epoch
    steps_per_epoch = len(train_features) // args.batch_size
    validation_steps = len(val_features) // args.batch_size
    
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    if validation_steps == 0:
        validation_steps = 1
    
    # Define the model
    print("Creating model...")
    if args.advanced_model:
        model = define_captioning_model(vocab_size, max_length)
    else:
        model = define_model(vocab_size, max_length)
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(args.model_dir, 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        verbose=1,
        min_lr=0.00001
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train the model
    print(f"Training model for {args.epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(args.model_dir, 'final_model.h5'))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, 'training_history.png'))
    plt.close()
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN-LSTM model for video captioning")
    parser.add_argument('--captions_path', type=str, default='data/captions.txt', help='Path to captions file')
    parser.add_argument('--images_dir', type=str, default='data/Images', help='Directory containing images')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save model and artifacts')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=8, help='Patience for early stopping')
    parser.add_argument('--advanced_model', action='store_true', help='Use advanced model architecture')
    
    args = parser.parse_args()
    train(args) 