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

# import tensorflow as tf

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

# print(train_data.class_names)

# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.1),
#     tf.keras.layers.RandomZoom(0.1)
# ])

# model = tf.keras.Sequential([
#     data_augmentation,
#     tf.keras.layers.Rescaling(1./255),

#     tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
#     tf.keras.layers.MaxPooling2D(),

#     tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
#     tf.keras.layers.MaxPooling2D(),

#     tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
#     tf.keras.layers.MaxPooling2D(),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(128,activation="relu"),
#     tf.keras.layers.Dropout(0.5),

#     tf.keras.layers.Dense(1,activation="sigmoid")
# ])

# model.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     "cat_dog_model.keras",
#     save_best_only=True,
#     monitor="val_accuracy"
# )

# model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=300,
#     callbacks=[checkpoint]
# )

# print("Model Saved Successfully")

import tensorflow as tf

IMG_SIZE = (128, 128)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=IMG_SIZE,
    batch_size=8,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=IMG_SIZE,
    batch_size=8,
    validation_split=0.2,
    subset="validation",
    seed=42
)

print(train_data.class_names)

# Cache + prefetch for speed
train_data = train_data.cache().prefetch(tf.data.AUTOTUNE)
val_data = val_data.cache().prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
])

# --- Pretrained base -----------------------------------------------------
# MobileNetV2 already knows general visual features (edges, textures, shapes)
# from being trained on 1.4M+ images. We freeze it and only train a small
# classifier head on top -- this is what makes a tiny dataset workable.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze pretrained weights

model = tf.keras.Sequential([
    data_augmentation,
    # MobileNetV2 expects inputs scaled to [-1, 1]. This does the same math
    # as mobilenet_v2.preprocess_input (x/127.5 - 1), but as a native
    # Rescaling layer instead of Lambda -- Lambda layers that wrap an
    # external function can't be reliably saved/reloaded in a .keras file,
    # which is what caused the "could not locate function" error.
    tf.keras.layers.Rescaling(scale=1./127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "cat_dog_model.keras",
    save_best_only=True,
    monitor="val_accuracy"
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[checkpoint, early_stop]
)

print("Model Saved Successfully")