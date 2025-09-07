'''
Author: Cody Costa
Date:   9/5/2025

'''

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.applications.efficientnet import EfficientNetB0
# model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(128,128,3))


# ====================================================
# 1. Dataset Preprocessing
# ====================================================
dataset_dir = f'{os.path.expanduser('~')}\\Downloads\\archive\\raw-img'

datagen_train = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True    # lighter augmentation for speed
)

datagen_val = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen_train.flow_from_directory(
    dataset_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='sparse',
    subset='training',
    color_mode='rgb'    # force 3 channels
)

val_gen = datagen_val.flow_from_directory(
    dataset_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    color_mode='rgb'    # force 3 channels
)

print("Class mapping:", train_gen.class_indices)

batch_X, batch_y = next(train_gen)
print(batch_X.shape)  # should be (batch, 128, 128, 3)  looks good

input('pause: ')

# ====================================================
# 2. Build Model (EfficientNetB0 backbone)
# ====================================================
# base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128,128,3))

base_model = EfficientNetB0(
    weights=None,   # no pretrained weights
    include_top=False,
    input_shape=(128,128,3)
)

base_model.trainable = False  # Stage 1: freeze backbone

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
output = Dense(6, activation='softmax')(x)   # 6 classes

model = Model(inputs=base_model.input, outputs=output)

# Compile for Stage 1
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ====================================================
# 3. Callbacks
# ====================================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2)

# ====================================================
# 4. Stage 1 Training (frozen backbone)
# ====================================================
print("\n--- Stage 1: Training classifier head only ---\n")
history_stage1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ====================================================
# 5. Stage 2 Fine-Tuning (unfreeze top layers)
# ====================================================
print("\n--- Stage 2: Fine-tuning EfficientNetB0 top layers ---\n")

base_model.trainable = True

# Freeze lower layers, unfreeze top ~50 for fine-tuning
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompile with a lower LR
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_stage2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ====================================================
# 6. Evaluation
# ====================================================
loss, acc = model.evaluate(val_gen, verbose=1)
print(f"Final Validation Accuracy: {acc:.3f}")
