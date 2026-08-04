import tensorflow as tf

model = tf.keras.models.load_model("cat_dog_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("cat_dog_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved cat_dog_model.tflite")