import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import string
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords')

def load_captions(captions_file):
    """Load and preprocess captions"""
    df = pd.read_csv(captions_file)
    # Remove header if it exists
    if df.columns[0] == 'image' and df.columns[1] == 'caption':
        df = df.iloc[1:]
        df.columns = ['image', 'caption']
    
    # Create a dictionary mapping image names to captions
    captions_dict = {}
    for _, row in df.iterrows():
        image_name = row[0]
        caption = row[1]
        if image_name not in captions_dict:
            captions_dict[image_name] = []
        captions_dict[image_name].append(caption)
    
    return captions_dict

def clean_caption(caption):
    """Clean and preprocess caption text"""
    # Convert to lowercase
    caption = caption.lower()
    
    # Remove punctuation
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    
    # Remove single character words and stopwords
    words = caption.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if len(word) > 1 and word not in stop_words]
    
    # Add start and end tokens
    caption = 'startseq ' + ' '.join(words) + ' endseq'
    
    return caption

def preprocess_captions(captions_dict):
    """Preprocess all captions"""
    cleaned_captions = {}
    for image_name, captions_list in captions_dict.items():
        cleaned_captions[image_name] = [clean_caption(caption) for caption in captions_list]
    
    return cleaned_captions

def create_tokenizer(captions_dict):
    """Create tokenizer from captions"""
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    
    return tokenizer

def max_caption_length(captions_dict):
    """Find the maximum caption length"""
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    
    return max(len(caption.split()) for caption in all_captions)

def extract_features(image_path, model):
    """Extract features from image using InceptionV3"""
    try:
        img = Image.open(image_path)
        img = img.resize((299, 299))
        img = np.array(img)
        
        # Handle grayscale images
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        # Handle RGBA images
        elif img.shape[2] == 4:
            img = img[:, :, :3]
            
        # Preprocess the image for InceptionV3
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        features = model.predict(img, verbose=0)
        return features
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def load_image_features(image_dir, feature_model, images_list=None):
    """Extract features from all images in directory"""
    features = {}
    
    if images_list is None:
        images_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Extracting features from {len(images_list)} images...")
    for image_name in tqdm(images_list):
        image_path = os.path.join(image_dir, image_name)
        image_features = extract_features(image_path, feature_model)
        
        if image_features is not None:
            features[image_name] = image_features
    
    return features

def create_feature_extractor():
    """Create feature extractor model"""
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)  # Remove classification layer
    return model

def create_sequences(tokenizer, max_length, captions_dict, features, vocab_size):
    """Create sequences for training"""
    X1, X2, y = [], [], []
    
    for image_name, captions in captions_dict.items():
        if image_name in features:
            feature = features[image_name][0]
            
            for caption in captions:
                # Convert caption to sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                
                # Create input-output pairs for different lengths (teacher forcing)
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    
                    # Pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    
                    # One-hot encode output sequence (fixed)
                    out_seq = np.zeros(vocab_size)
                    out_seq[out_seq] = 1
                    
                    # Add to training data
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
    
    return np.array(X1), np.array(X2), np.array(y)

def create_data_generator(tokenizer, max_length, captions_dict, features, vocab_size, batch_size=32):
    """Create a generator for training data"""
    image_names = list(features.keys())
    indices = list(range(len(image_names)))
    
    while True:
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            current_indices = indices[i:min(i + batch_size, len(indices))]
            
            X1, X2, y = [], [], []
            
            for idx in current_indices:
                image_name = image_names[idx]
                
                if image_name in captions_dict:
                    feature = features[image_name][0]
                    captions = captions_dict[image_name]
                    
                    # Select a random caption for this image
                    caption = np.random.choice(captions)
                    
                    # Convert caption to sequence
                    seq = tokenizer.texts_to_sequences([caption])[0]
                    
                    # Generate input-output pairs
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        
                        # Pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        
                        # One-hot encode output (fixed)
                        out_seq_vec = np.zeros(vocab_size)
                        out_seq_vec[out_seq] = 1  # Correct index for one-hot
                        
                        X1.append(feature)
                        X2.append(in_seq)
                        y.append(out_seq_vec)
            
            if X1:
                yield [np.array(X1), np.array(X2)], np.array(y)

def word_for_id(integer, tokenizer):
    """Map an integer to a word"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None 