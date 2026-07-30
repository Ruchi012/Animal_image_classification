import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("cat_dog_model.keras")

img = tf.keras.preprocessing.image.load_img(
     "test.jpg",
     target_size=(128,128)
 )

img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Dog")
else:
    print("Cat")