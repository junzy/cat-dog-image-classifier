import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Load the dataset
ds, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

# `ds` is a dictionary with keys 'train', 'test', (and possibly others like 'validation') depending on the dataset.
# `info` contains information about the dataset, such as the number of samples.

# Initialize counters for each label
label_counts = {0: 0, 1: 0}

# Count the number of images for each label
for image, label in ds['train']:
    label_counts[label.numpy()] += 1

print(f"Total images of cats: {label_counts[0]}")
print(f"Total images of dogs: {label_counts[1]}")

# Display a larger number of cats and dogs
plt.figure(figsize=(15, 10))
for i, (image, label) in enumerate(ds['train'].take(16)):  # Display the first 16 images
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(image.numpy())
    plt.title('Cat' if label.numpy() == 0 else 'Dog')
    plt.axis('off')
plt.show()
