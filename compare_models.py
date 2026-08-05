import tensorflow as tf
import numpy as np

img = tf.keras.preprocessing.image.load_img("test.jpg", target_size=(128,128))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

# Original Keras model
keras_model = tf.keras.models.load_model("cat_dog_model.keras")
keras_score = keras_model.predict(img)[0][0]
print("Keras raw score:", keras_score)

# TFLite model
interpreter = tf.lite.Interpreter(model_path="cat_dog_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]["index"], img.astype(np.float32))
interpreter.invoke()
tflite_score = interpreter.get_tensor(output_details[0]["index"])[0][0]
print("TFLite raw score:", tflite_score)