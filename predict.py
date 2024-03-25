import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model (assuming it's saved as 'model')
# If the model is not saved, you can skip this step if you're running this immediately after training in the same script
# model = tf.keras.models.load_model('path_to_my_model.h5')

# Load and preprocess an image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.0


# Replace 'path_to_new_image.jpg' with the path to the image you want to classify
img_to_predict = load_and_preprocess_image('test.jpg')

model = tf.keras.models.load_model('saved_models/cats_vs_dogs_model.h5')
# Make a prediction
predictions = model.predict(img_to_predict)

# Assuming your model's final layer is a softmax activation, predictions will be a 2-element array of probabilities
# The index with the highest probability corresponds to the model's prediction
predicted_class = np.argmax(predictions, axis=1)

# Interpret the prediction
if predicted_class[0] == 0:
    print("It's a cat!")
else:
    print("It's a dog!")
