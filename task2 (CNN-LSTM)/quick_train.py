import os
import pickle
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from utils import (
    load_captions, preprocess_captions, create_tokenizer, 
    max_caption_length, create_feature_extractor, 
    load_image_features, create_data_generator
)
from model import define_model

def quick_train(captions_path='data/captions.txt', 
                images_dir='data/Images', 
                model_dir='models',
                max_samples=500,  # Increased from 100 to 500
                batch_size=32,
                epochs=10):  # Increased from 3 to 10
    """Train the CNN-LSTM model with a subset of data for testing"""
    
    # Create output directories if they don't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print("Loading and preprocessing captions...")
    captions_dict = load_captions(captions_path)
    
    # Preprocess captions
    captions_dict = preprocess_captions(captions_dict)
    
    # Limit to max_samples
    image_names = list(captions_dict.keys())
    np.random.shuffle(image_names)
    selected_images = image_names[:max_samples]
    selected_captions = {img: captions_dict[img] for img in selected_images if img in captions_dict}
    
    print(f"Selected {len(selected_captions)} images for quick training")
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(selected_captions)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")
    
    # Find maximum caption length
    max_length = max_caption_length(selected_captions)
    print(f"Maximum Caption Length: {max_length}")
    
    # Save tokenizer and max_length for later use
    with open(os.path.join(model_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(model_dir, 'max_length.pkl'), 'wb') as f:
        pickle.dump(max_length, f)
    
    # Create feature extractor (CNN) model
    print("Creating feature extractor...")
    feature_extractor = create_feature_extractor()
    
    # Extract features only from selected images
    print("Extracting features from selected images...")
    features = load_image_features(images_dir, feature_extractor, selected_images)
    
    # Save features for later use
    with open(os.path.join(model_dir, 'features.pkl'), 'wb') as f:
        pickle.dump(features, f)
    
    # Calculate dataset sizes
    num_images = len(features)
    train_size = int(num_images * 0.8)
    
    # Get list of image names
    image_names = list(features.keys())
    np.random.shuffle(image_names)
    
    # Split into training and validation sets
    train_images = image_names[:train_size]
    val_images = image_names[train_size:]
    
    # Create dictionaries for training and validation sets
    train_features = {img: features[img] for img in train_images if img in features}
    val_features = {img: features[img] for img in val_images if img in features}
    
    train_captions = {img: selected_captions[img] for img in train_images if img in selected_captions}
    val_captions = {img: selected_captions[img] for img in val_images if img in selected_captions}
    
    # Create data generators
    print("Creating data generators...")
    train_generator = create_data_generator(
        tokenizer, max_length, train_captions, train_features, 
        vocab_size, batch_size=batch_size
    )
    val_generator = create_data_generator(
        tokenizer, max_length, val_captions, val_features, 
        vocab_size, batch_size=batch_size
    )
    
    # Determine steps per epoch (ensure at least 1 step)
    steps_per_epoch = max(1, len(train_features) // batch_size)
    validation_steps = max(1, len(val_features) // batch_size)
    
    # Define the model
    print("Creating model...")
    model = define_model(vocab_size, max_length)
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'quick_model-ep{epoch:02d}.h5'),
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,  # Small patience for quick training
        verbose=1,
        restore_best_weights=True
    )
    
    callbacks = [checkpoint, early_stopping]
    
    # Train the model
    print(f"Training model for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(model_dir, 'final_model.h5'))
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'quick_training_history.png'))
    plt.close()
    
    print("Quick training completed!")

if __name__ == "__main__":
    quick_train() 