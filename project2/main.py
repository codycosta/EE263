import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



# 1. Load the .pkl dataset
data = pd.read_pickle('waferImg26x26.pkl')

# 2. Extract images and labels
if isinstance(data, pd.DataFrame):
    images_list = data['images'].values
    y = data['labels'].values

elif isinstance(data, dict):
    images_list = data['images']
    y = np.array(data['labels'])

else:
    raise ValueError("Unexpected pickle format")

# 3. Process images
processed_images = []

for img in images_list:
    img = np.array(img)

    # Convert channel-first to channel-last
    if img.ndim == 3 and img.shape[0]== 3:  # (C,H,W) -> (H,W,C)
        img = np.transpose(img, (1,2,0))

    # Convert grayscale to RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    # Resize to 26x26
    img = cv2.resize(img, (26,26))
    processed_images.append(img)

X = np.stack(processed_images).astype('float32')/ 255.0  # Normalize

# 4. Process labels: flatten nested labels and map to integers
y_flat = []

for label in y:
    if isinstance(label, (list,np.ndarray)):
        y_flat.append(label[0])  # takefirst element if nested

    else:
        y_flat.append(label)

# Map unique labels to integers
unique_labels = sorted(set(y_flat))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
y_int = np.array([label_to_int[label]for label in y_flat])

num_classes = len(unique_labels)
y = to_categorical(y_int, num_classes)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# 6. Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(26,26,3)), 
    MaxPooling2D((2,2)), 
    Conv2D(64, (3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 7. Train
history = model.fit(X_train,y_train, epochs=20, batch_size=32, validation_split=0.1)

# 8. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy:{test_acc:.4f}")