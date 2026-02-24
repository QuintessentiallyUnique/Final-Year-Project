import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow_datasets as tfds

# Configuration
IMG_SIZE = (192, 192)
BATCH_SIZE = 32
EPOCHS = 100
NUM_FOLDS = 5
SELECTED_CLASSES = [
    'chicken_curry', 'grilled_salmon', 'pizza', 'samosa', 'garlic_bread',
    'steak', 'french_fries', 'hamburger', 'spaghetti_bolognese', 'falafel'
]

# Load Data
def load_and_preprocess_data():
    ds_train, info = tfds.load('food101', split='train', with_info=True, as_supervised=True)
    ds_val = tfds.load('food101', split='validation', as_supervised=True)
    
    full_ds = ds_train.concatenate(ds_val)
    
    images = []
    labels = []
    
    all_class_names = info.features['label'].names
    selected_indices = [all_class_names.index(c) for c in SELECTED_CLASSES]
    
    for img, label in full_ds:
        if label in selected_indices:
            new_label = selected_indices.index(label)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
            images.append(img.numpy())
            labels.append(new_label)
            
    return np.array(images), np.array(labels)

X_full, y_full = load_and_preprocess_data()
print(f"Total Images: {len(X_full)}")

X_cv = X_full
y_cv = y_full

# Augmentation
def mixup(image, label):
    alpha = 0.2
    batch_size = tf.shape(image)[0]
    
    l = tf.random.gamma([batch_size, 1], alpha)
    l = l / (l + tf.random.gamma([batch_size, 1], alpha))

    l_img = tf.reshape(l, [batch_size, 1, 1, 1])
    
    indices = tf.range(batch_size)
    shuffled_indices = tf.random.shuffle(indices)
    
    image_a = image
    image_b = tf.gather(image, shuffled_indices)
    label_a = label
    label_b = tf.gather(label, shuffled_indices)
    
    new_image = l_img * image_a + (1 - l_img) * image_b
    new_label = l * label_a + (1 - l) * label_b
    
    return new_image, new_label

def prepare_dataset(X, y, is_training=True):
    y = tf.keras.utils.to_categorical(y, num_classes=10)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    
    if is_training:
        ds = ds.shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(mixup, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(BATCH_SIZE)
        
    return ds.prefetch(tf.data.AUTOTUNE)

def build_model():
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    
    def conv_block(x, filters, kernel_size=3):
        # Conv 1
        x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False,
                          kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(1e-3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        # Conv 2
        x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False,
                          kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(1e-3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        # Max Pool
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        return x

    x = conv_block(inputs, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)

    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="wide_vgg_sgd")
    
    total_steps = EPOCHS * (len(X_cv) // BATCH_SIZE // NUM_FOLDS * (NUM_FOLDS-1))
    
    lr_scheduler = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.01, decay_steps=total_steps, alpha=0.01
    )
    
    optimizer = keras.optimizers.SGD(
        learning_rate=lr_scheduler, 
        momentum=0.9, 
        nesterov=True
    )
    
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model

# 5-Fold CV
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

print(f"\nStarting {NUM_FOLDS}-Fold Cross Validation")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
    print(f"\nFold {fold+1}/{NUM_FOLDS}")
    
    # Split
    X_train, X_val = X_cv[train_idx], X_cv[val_idx]
    y_train, y_val = y_cv[train_idx], y_cv[val_idx]
    
    train_ds = prepare_dataset(X_train, y_train, is_training=True)
    val_ds = prepare_dataset(X_val, y_val, is_training=False)
    
    model = build_model()
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=50, restore_best_weights=True, verbose=1
    )
    
    log_file = f"fold_{fold+1}_log.csv"
    csv_logger = keras.callbacks.CSVLogger(log_file, append=False)
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, 
              callbacks=[early_stop, csv_logger], verbose=1)
    
    _, acc = model.evaluate(val_ds, verbose=0)
    print(f"Fold {fold+1} Accuracy: {acc:.2%}")
    
    model.save(f"model_fold_{fold+1}.keras")
    
    tf.keras.backend.clear_session()
    del model, train_ds, val_ds
    gc.collect()
    