# Larq Zoo Examples

## Classify ImageNet classes with Bi-Real Net

```python
import tensorflow as tf
import numpy as np
import larq_zoo as lqz

model = lqz.BiRealNet(weights="imagenet")

img_path = "tests/fixtures/elephant.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = lqz.preprocess_input(x)
x = np.expand_dims(x, axis=0)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print("Predicted:", lqz.decode_predictions(preds, top=3)[0])
# Predicted: [("n01871265", "tusker", 0.7427464), ("n02504458", "African_elephant", 0.19439144), ("n02504013", "Indian_elephant", 0.058899447)]
```

## Extract features with Bi-Real Net

```python
import tensorflow as tf
import numpy as np
import larq_zoo as lqz

model = lqz.BiRealNet(weights="imagenet", include_top=False)

img_path = "tests/fixtures/elephant.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = lqz.preprocess_input(x)
x = np.expand_dims(x, axis=0)

features = model.predict(x)
```

## Extract features from an arbitrary intermediate layer with Bi-Real Net

```python
import tensorflow as tf
import numpy as np
import larq_zoo as lqz

base_model = lqz.BiRealNet(weights="imagenet")
model = tf.keras.models.Model(
    inputs=base_model.input, outputs=base_model.get_layer("average_pooling2d_8").output
)

img_path = "tests/fixtures/elephant.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = lqz.preprocess_input(x)
x = np.expand_dims(x, axis=0)

average_pool_8_features = model.predict(x)
```

## Fine-tune Bi-Real Net on a new set of classes

```python
import tensorflow as tf
import larq as lq
import larq_zoo as lqz

# create the base pre-trained model
base_model = lqz.BiRealNet(weights="imagenet", include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# let's add a binarized fully-connected layer
x = lq.layers.QuantDense(
    1024,
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip",
    use_bias=False,
    activation="relu",
)(x)
x = tf.keras.layers.BatchNormalization()(x)
# and a full precision logistic layer -- let's say we have 200 classes
predictions = tf.keras.layers.Dense(200, activation="softmax")(x)

# this is the model we will train
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Bi-Real Net layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

# train the model on the new data for a few epochs
model.fit(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from Bi-Real Net. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top block, i.e. we will freeze
# the first 49 layers and unfreeze the rest:
for layer in model.layers[:49]:
   layer.trainable = False
for layer in model.layers[49:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
    loss="categorical_crossentropy",
)

# we train our model again (this time fine-tuning the top block
# alongside the top Dense layers
model.fit(...)
```

### Build Bi-Real Net over a custom input Tensor

```python
import tensorflow as tf
import larq_zoo as lqz

# this could also be the output a different Keras model or layer
input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))

model = lqz.BiRealNet(input_tensor=input_tensor, weights="imagenet")
```

## Evaluate Bi-Real Net with TensorFlow Datasets

```python
import tensorflow_datasets as tfds
import larq_zoo as lqz
import tensorflow as tf


def preprocess(data):
    return lqz.preprocess_input(data["image"]), tf.one_hot(data["label"], 1000)

dataset = (
    tfds.load("imagenet2012", split=tfds.Split.VALIDATION)
    .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(128)
    .prefetch(1)
)

model = lqz.BiRealNet()
model.compile(
    optimizer="sgd",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
)

model.evaluate(dataset)
```
