import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.stats import entropy

class InceptionScore:
    def __init__(self):
        self.model = InceptionV3(include_top=True, weights='imagenet', input_shape=(299, 299, 3))

    # Load pretrained InceptionV3 model

    def load_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if img is not None:
                    images.append(img)
        return images

    def calculate_inception_score(self, images, batch_size=32):
        # Resize and preprocess images
        processed_images = [preprocess_input(image.img_to_array(img.resize((299, 299))))
                            for img in images]

        # Predict class probabilities
        pred = self.model.predict(np.array(processed_images), batch_size=batch_size)
        
        # Calculate marginal distribution
        marginal_dist = np.mean(pred, axis=0)

        # Calculate KL divergence and Inception Score
        kl_div = pred * (np.log(pred + 1e-16) - np.log(np.expand_dims(marginal_dist, 0)))
        kl_div_mean = np.mean(np.sum(kl_div, axis=1))
        
        inception_score = np.exp(kl_div_mean)
        
        return inception_score



