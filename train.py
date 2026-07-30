# import tensorflow as tf

# # Load Dataset
# train_data = tf.keras.preprocessing.image_dataset_from_directory(
#     "dataset",
#     image_size=(128,128),
#     batch_size=8,
#     validation_split=0.2,
#     subset="training",
#     seed=42
# )

# val_data = tf.keras.preprocessing.image_dataset_from_directory(
#     "dataset",
#     image_size=(128,128),
#     batch_size=8,
#     validation_split=0.2,
#     subset="validation",
#     seed=42
# )

# # CNN Model
# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1./255),

#     tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
#     tf.keras.layers.MaxPooling2D(),

#     tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
#     tf.keras.layers.MaxPooling2D(),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(128,activation="relu"),

#     tf.keras.layers.Dense(1,activation="sigmoid")
# ])

# model.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# model.fit(train_data, validation_data=val_data, epochs=5)

# model.save("cat_dog_model.keras")

# print("Model Saved Successfully")

import tensorflow as tf

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(128,128),
    batch_size=8,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(128,128),
    batch_size=8,
    validation_split=0.2,
    subset="validation",
    seed=42
)

print(train_data.class_names)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "cat_dog_model.keras",
    save_best_only=True,
    monitor="val_accuracy"
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[checkpoint]
)

print("Model Saved Successfully")