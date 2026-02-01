import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from dataset import get_dataset, SELECTED_CLASSES

IMG_SIZE = (192, 192)
BATCH_SIZE = 32
EPOCHS = 150 
MODEL_OUT = "cnn_scratch.keras"

train_ds, val_ds, _, class_names = get_dataset(
    selected_classes=SELECTED_CLASSES,
    batch_size=BATCH_SIZE
)
num_classes = len(class_names)
print(f"Classes: {class_names}")

def conv_block(x, filters, kernel_size=3):
    # Conv 1 with L2 Regularization
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Conv 2 with L2 Regularization
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Max Pool
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    return x

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)

x = conv_block(x, 64)
x = conv_block(x, 128)
x = conv_block(x, 256)
x = conv_block(x, 512)

x = layers.GlobalMaxPooling2D()(x)
x = layers.Dropout(0.6)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="wide_vgg_regularized")
model.summary()


total_train_images = num_classes * 750
steps_per_epoch = math.ceil(total_train_images / BATCH_SIZE)
decay_steps = EPOCHS * steps_per_epoch

print(f"Total training images: {total_train_images}")
print(f"Steps per epoch: {steps_per_epoch}")

lr_scheduler = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=decay_steps,
    alpha=0.01
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_OUT,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.CSVLogger("training_log.csv", append=False)
]

#Train
print("\nStarting Training")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print("\nLoading best weights for TTA...")
model.load_weights(MODEL_OUT)

val_images = []
val_labels = []
for imgs, lbls in val_ds:
    val_images.append(imgs)
    val_labels.append(lbls)
val_images = tf.concat(val_images, axis=0)
val_labels = tf.concat(val_labels, axis=0)

# Predict original + flipped
pred_1 = model.predict(val_images, batch_size=BATCH_SIZE, verbose=1)
pred_2 = model.predict(tf.image.flip_left_right(val_images), batch_size=BATCH_SIZE, verbose=1)

# Average them
final_pred = (pred_1 + pred_2) / 2.0
final_labels = np.argmax(final_pred, axis=1)

# Convert one-hot ground truth back to integers
true_labels = np.argmax(val_labels.numpy(), axis=1)

acc = np.mean(final_labels == true_labels)
print("\n" + "="*40)
print(f"FINAL TTA ACCURACY: {acc:.2%}")
print("="*40)