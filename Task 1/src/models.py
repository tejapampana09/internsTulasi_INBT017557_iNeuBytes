import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def get_baseline_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Creates a clean baseline CNN (AlexNet-style adapted for small 32x32 images).
    Increasing filters (32 -> 64 -> 128), ReLU activations, pooling,
    and fully connected layers. No regularization, no batch norm, no dropout.
    """
    model = models.Sequential([
        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten
        layers.Flatten(),
        
        # FC Block
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name="Baseline_CNN")
    return model

def get_dropout_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Baseline model + Dropout (0.3 after pooling, 0.5 in FC layer)
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ], name="Regularization_Dropout")
    return model

def get_bn_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Baseline model + Batch Normalization after convolutions (before activation)
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # FC Block
        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(10, activation='softmax')
    ], name="Regularization_BatchNormalization")
    return model

def get_l2_model(input_shape=(32, 32, 3), num_classes=10, l2_val=1e-4):
    """
    Baseline model + L2 regularization (weight decay = 1e-4) on Conv/Dense layers
    """
    reg = regularizers.l2(l2_val)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg, input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=reg),
        layers.Dense(10, activation='softmax')
    ], name="Regularization_L2")
    return model

def get_augmentation_layers(level):
    """
    Returns Keras preprocessing layers for data augmentation
    """
    if level == 'light':
        return [
            layers.RandomFlip("horizontal")
        ]
    elif level == 'moderate':
        return [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomTranslation(0.1, 0.1)
        ]
    elif level == 'aggressive':
        return [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomTranslation(0.2, 0.2),
            layers.RandomZoom(0.2)
        ]
    return []

def get_augmentation_model(level, input_shape=(32, 32, 3), num_classes=10):
    """
    Baseline model + Data Augmentation layers
    """
    aug_layers = get_augmentation_layers(level)
    
    # We combine augmentation layers and baseline layers
    all_layers = aug_layers + [
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
    model = models.Sequential(all_layers, name=f"DataAugmentation_{level.capitalize()}")
    return model

def get_deeper_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Baseline model + 1 extra conv block (Conv2D 256)
    Architecture: 32 -> 64 -> 128 -> 256
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4 (Extra Block)
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # FC Block
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name="Architecture_Deeper")
    return model

def get_final_custom_model(best_features, input_shape=(32, 32, 3), num_classes=10):
    """
    Combines the best-performing components based on Part B experiments.
    Example configuration might include: BatchNormalization, Dropout, Moderate Augmentation, Adam optimizer.
    """
    model_layers = []
    
    # 1. Add Data Augmentation if it was helpful
    if 'augmentation' in best_features and best_features['augmentation'] is not None:
        model_layers.extend(get_augmentation_layers(best_features['augmentation']))
        
    # 2. Build core blocks with Conv, BN (if helpful), Activation, MaxPool, Dropout (if helpful)
    use_bn = best_features.get('use_bn', False)
    use_dropout = best_features.get('use_dropout', False)
    use_l2 = best_features.get('use_l2', False)
    use_deeper = best_features.get('use_deeper', False)
    
    reg = regularizers.l2(1e-4) if use_l2 else None
    
    # Block 1: 32 filters
    model_layers.append(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg, input_shape=input_shape))
    if use_bn:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation('relu'))
    model_layers.append(layers.MaxPooling2D((2, 2)))
    if use_dropout:
        model_layers.append(layers.Dropout(0.3))
        
    # Block 2: 64 filters
    model_layers.append(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg))
    if use_bn:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation('relu'))
    model_layers.append(layers.MaxPooling2D((2, 2)))
    if use_dropout:
        model_layers.append(layers.Dropout(0.3))
        
    # Block 3: 128 filters
    model_layers.append(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg))
    if use_bn:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation('relu'))
    model_layers.append(layers.MaxPooling2D((2, 2)))
    if use_dropout:
        model_layers.append(layers.Dropout(0.3))
        
    # Block 4: Deeper (if helpful)
    if use_deeper:
        model_layers.append(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg))
        if use_bn:
            model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.Activation('relu'))
        model_layers.append(layers.MaxPooling2D((2, 2)))
        if use_dropout:
            model_layers.append(layers.Dropout(0.3))
            
    # Flatten and Dense Block
    model_layers.append(layers.Flatten())
    
    dense_units = 256 if use_deeper else 128
    model_layers.append(layers.Dense(dense_units, kernel_regularizer=reg))
    if use_bn:
        model_layers.append(layers.BatchNormalization())
    model_layers.append(layers.Activation('relu'))
    if use_dropout:
        model_layers.append(layers.Dropout(0.5))
        
    # Output Layer
    model_layers.append(layers.Dense(num_classes, activation='softmax'))
    
    model = models.Sequential(model_layers, name="Final_Customized_CNN")
    return model
