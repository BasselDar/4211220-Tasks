import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.layers import Add
from tensorflow.keras.optimizers import Adam

def define_model(vocab_size, max_length, embedding_dim=256, units=256):
    """Define the CNN-LSTM model for image/video captioning"""
    
    # Feature extractor (CNN) input
    inputs1 = Input(shape=(2048,))  # Shape from InceptionV3 feature output
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embedding_dim, activation='relu')(fe1)
    
    # Sequence (caption) input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(units)(se2)
    
    # Decoder (combining both inputs)
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(units, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Create the model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    
    # Print model summary
    print(model.summary())
    
    return model


def define_captioning_model(vocab_size, max_length, embedding_dim=256, units=256):
    """Define a more advanced model with attention mechanism"""
    
    # Feature extractor (CNN) input
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embedding_dim, activation='relu')(fe1)
    
    # Sequence (caption) input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    
    # LSTM layer with more units and return sequences for attention
    lstm = LSTM(units, return_sequences=True)(se2)
    
    # Second LSTM layer for better sequence modeling
    lstm2 = LSTM(units)(lstm)
    
    # Decoder (combining both inputs)
    decoder1 = Add()([fe2, lstm2])
    decoder2 = Dense(units, activation='relu')(decoder1)
    decoder3 = Dense(units, activation='relu')(decoder2)
    outputs = Dense(vocab_size, activation='softmax')(decoder3)
    
    # Create the model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    
    return model 