from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import requests
import cv2
from io import BytesIO







def fetch_Image(category, access_code = 'InH17fKA7WlcADcSpzeSLZoPpjvRL7Ms8jRfgRSybHY', max_pages=15):
    base_url = 'https://api.unsplash.com/'

    query = category
    image_urls = []
    for page in range(1, max_pages + 1):
        response = requests.get(base_url + f'search/photos?query={query}&page={page}&per_page=10', headers={'Authorization': f'Client-ID {access_code}'})
        
        if response.status_code == 200:
            
            data = response.json()
            page_urls = [photo['urls']['regular'] for photo in data['results']]
            image_urls.extend(page_urls)
            if page >= data['total_pages']:
                break  
        else:
            print('Error fetching images from Unsplash:', response.status_code)
            break
    return image_urls


def Image_URL(url, target_size=(64,64)):
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content) , dtype=np.uint8)
    image = cv2.imdecode(image_array , cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = cv2.resize(image, target_size) / 255.0  
    return image

def build_generator(latent_dim):
    model = tf.keras.Sequential ([
        layers.Dense(8*8*128,activation = 'relu', input_dim= latent_dim),
        layers.Reshape((8,8,128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')
        
    ])
    return model

def build_discriminator(inputShape):
    model = tf.keras.Sequential ([
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=inputShape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
        
    ])
    return model

def build_gan(generator , discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    discriminator.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1= 0.5),loss = 'binary_crossentropy')

    return model



def train_generator(gan , noise):
    NP = np.ones((noise.shape[0],1))
    gan.train_on_batch(noise,NP)

def trainDiscriminator(discriminator , real_images , fake_images) : 
    inputs = np.concatenate([real_images,fake_images])
    labels = np.concatenate([np.ones((real_images.shape[0], 1)) * 0.9, np.zeros((fake_images.shape[0], 1))])
    discriminator.train_on_batch(inputs, labels)
    

def preprocess(image_urls , targetSize =(64,64)):
    image_array =[Image_URL(url, targetSize) for url in image_urls]
    return np.array(image_array)

def genrate_noise(batchSize , latent_dim):
    return np.random.normal(0,1, (batchSize, latent_dim))


    

imageName = input("Please tell the object name:")
print(f"Generating image of {imageName}")
imageURL = fetch_Image(imageName)
imageArray = preprocess(imageURL)
#print(imageArray.shape)

latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator((64,64,3))
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

gan = build_gan(generator,discriminator)
epochs = 5000
batch_size = 32

for epoch in range(epochs):
    real_images = imageArray[np.random.randint(0, imageArray.shape[0], batch_size)]
    noise = genrate_noise(batch_size, latent_dim)
    fake_images = generator.predict(noise)
    trainDiscriminator(discriminator, real_images, fake_images)

    # Train the generator via GAN
    noise = genrate_noise(batch_size, latent_dim)
    train_generator(gan, noise)

    # Printing progress and occasionally visualize
"""
    if epoch % 1000 == 0:  # Adjust this interval as needed
        print(f"Epoch: {epoch}")
        noise = genrate_noise(1, latent_dim)
        outputImage = generator.predict(noise)[0]
        outputImage = (outputImage * 255).astype(np.uint8)
        cv2.imshow('Generated Image', outputImage)
        cv2.waitKey(1)

noise = genrate_noise(1, latent_dim)


trainDiscriminator(discriminator, imageArray, generator.predict(noise))
train_generator(generator ,discriminator ,noise)
"""
final_noise = genrate_noise(1, latent_dim)
OutputImage = generator.predict(final_noise)[0]
OutputImage = (OutputImage * 255).astype(np.uint8)

cv2.imshow('Generated Image' , OutputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

#/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 /Users/siraaj/python.py/GAN.py








