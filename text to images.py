#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import hashlib
import matplotlib.pyplot as plt
from keras_cv.models import StableDiffusion

def generate_birdfeeder_images_from_text(seed_text):
    # Set up the StableDiffusion model
    model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)

    # Set up parameters for generation
    num_images = 5
    batch_size = 1 
    num_steps = 250

    # Convert seed_text to an integer using a hash function
    seed = int(hashlib.sha256(seed_text.encode('utf-8')).hexdigest(), 16) % 10**8

    # Set the random seed for reproducibility
    np.random.seed(seed=seed)

    generated_images = []
    for _ in range(num_images):
        # Generate an image based on the given text
        generated_image = model.text_to_image(
            prompt=seed_text,
            batch_size=batch_size,
            num_steps=num_steps,
            seed=np.random.randint(0, 10000))
        generated_images.append(generated_image[0])

    return generated_images

# Example usage
seed_input = "a blue bird, squirrel fighting near a birdfeeder"
generated_images = generate_birdfeeder_images_from_text(seed_input)

# Display the generated images
fig, axes = plt.subplots(1, len(generated_images))
for i, image in enumerate(generated_images):
    axes[i].imshow(image)
    axes[i].axis('off')

plt.show()


# In[2]:


# Example usage

# Save the generated images
for i, image in enumerate(generated_images):
    filename = f"generated_image_{i}.png"
    plt.imsave(filename, image)

print("Images saved successfully.")


# In[ ]:




