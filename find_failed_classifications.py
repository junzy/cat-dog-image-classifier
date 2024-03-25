import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def preprocess(image, label):
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0
    return image, label

# Load and preprocess the dataset
(train_data, validation_data), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

train_data = train_data.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Load the model
model = tf.keras.models.load_model('saved_models/cats_vs_dogs_model.h5')

# Predict on the entire training dataset
predictions = model.predict(train_data)

# Convert softmax outputs to predicted class indices
predicted_classes = np.argmax(predictions, axis=1)

# To find misclassified examples, we need to iterate over the dataset again
misclassified_images = []
actual_labels = []
for images, labels in train_data.unbatch().batch(1).take(len(predictions)):
    actual_labels.append(labels.numpy()[0])

# Compare actual labels with predicted classes
for i, (image, label) in enumerate(train_data.unbatch().take(len(predictions))):
    if actual_labels[i] != predicted_classes[i]:
        misclassified_images.append((image.numpy(), actual_labels[i], predicted_classes[i]))
        if len(misclassified_images) >= 3:  # Stop after collecting 3 misclassified images
            break

# Display the misclassified images
for image, true_label, predicted_label in misclassified_images:
    plt.figure()
    plt.imshow(image)
    plt.title(f"True: {'Cat' if true_label == 0 else 'Dog'}, Predicted: {'Cat' if predicted_label == 0 else 'Dog'}")
    plt.axis('off')
    plt.show()
