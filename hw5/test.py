from keras.applications.efficientnet import EfficientNetB0
model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(128,128,3))
