# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
import glob
from tensorflow.keras.layers import Input, Dense,LeakyReLU,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from google.colab import drive
drive.mount("/content/drive")

x=[]
y=[]
k=[]

path="/content/drive/MyDrive/combined data/train/0"
img_files = os.listdir(path)
i=0
for img in  glob.glob("/content/drive/MyDrive/combined data/train/0/*.png"):
  img = cv.imread(path + '/' + img_files[i])
  x.append(img)
  y.append(0)
  i=i+1

path1="/content/drive/MyDrive/combined data/train/1"
img_files = os.listdir(path1)
i=0
for img in  glob.glob("/content/drive/MyDrive/combined data/train/1/*.png"):
  img = cv.imread(path1 + '/' + img_files[i])
  x.append(img)
  y.append(1)
  i=i+1

path2="/content/drive/MyDrive/combined data/train/2"
img_files = os.listdir(path2)
i=0
for img in  glob.glob("/content/drive/MyDrive/combined data/train/2/*.png"):
  img = cv.imread(path2 + '/' + img_files[i])
  x.append(img)
  y.append(2)
  i=i+1

path3="/content/drive/MyDrive/combined data/train/3"
img_files = os.listdir(path3)
i=0
for img in  glob.glob("/content/drive/MyDrive/combined data/train/3/*.png"):
  img = cv.imread(path3 + '/' + img_files[i])
  x.append(img)
  y.append(3)
  i=i+1

path4="/content/drive/MyDrive/combined data/train/4"
img_files = os.listdir(path4)
i=0
for img in  glob.glob("/content/drive/MyDrive/combined data/train/4/*.png"):
  img = cv.imread(path4 + '/' + img_files[i])
  x.append(img)
  y.append(4)
  i=i+1

path5="/content/drive/MyDrive/combined data/train/5"
img_files = os.listdir(path5)
i=0
for img in  glob.glob("/content/drive/MyDrive/combined data/train/5/*.png"):
  img = cv.imread(path5 + '/' + img_files[i])
  x.append(img)
  y.append(5)
  i=i+1

path6="/content/drive/MyDrive/combined data/train/6"
img_files = os.listdir(path6)
i=0
for img in  glob.glob("/content/drive/MyDrive/combined data/train/6/*.png"):
  img = cv.imread(path6 + '/' + img_files[i])
  x.append(img)
  y.append(6)
  i=i+1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
# Scale the inputs in range of (-1, +1) for better training
for i in range(0,len(x_train)):
  x_train[i]= x_train[i]/225 *2-1
for i in range(0,len(x_test)):
  x_test[i] = x_test[i]/225*2-1

x_train=np.array(x_train)
x_test=np.array(x_test)
N, H, W ,K= x_train.shape
print(x_train.shape)
D = H * W #(48*48)
x_train = x_train.reshape(-1, D)
x_test = x_test.reshape(-1, D)

# Defining Generator Model
latent_dim =100
def build_generator(latent_dim):
  i = Input(shape=(latent_dim,))
  x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
  x = BatchNormalization(momentum=0.7)(x)
  x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
  x = BatchNormalization(momentum=0.7)(x)
  x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
  x = BatchNormalization(momentum=0.7)(x)
  x = Dense(D, activation='tanh')(x)  #because Image pixel is between -1 to 1.
  model = Model(i, x)  #i is input x is output layer
  return model

def build_discriminator(img_size):
  i = Input(shape=(img_size,))
  x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
  x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
  x = Dense(1, activation='sigmoid')(x)
  model = Model(i, x)
  return model

# Build and compile the discriminator
discriminator = build_discriminator(D)
discriminator.compile ( loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])
# Build and compile the combined model
generator = build_generator(latent_dim)

z = Input(shape=(latent_dim,))
## Pass noise through a generator to get an Image
img = generator(z)
discriminator.trainable = False
fake_pred = discriminator(img)

combined_model_gen = Model(z, fake_pred)  #first is noise and 2nd is fake prediction
# Compile the combined model
combined_model_gen.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))

batch_size = 64
epochs = 12000
sample_period =200
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)
#store generator and discriminator loss in each step or each epoch
d_losses = []
g_losses = []
#create a file in which generator will create and save images
if not os.path.exists('/content/gan_images'):
  os.makedirs('/content/gan_images')

def sample_images(epoch):
  rows, cols = 5, 5
  noise = np.random.randn(rows * cols, latent_dim)
  imgs = generator.predict(noise)
  # Rescale images 0 - 1
  imgs = 0.5 * imgs + 0.5
  fig,axs = plt.subplots(rows, cols)  #fig to plot img and axis to store
  idx = 0
  for i in range(rows):  #5*5 loop means on page 25 imgs will be there
    for j in range(cols):
      axs[i,j].imshow(imgs[idx].reshape(H, W), cmap='gray')
      axs[i,j].axis('off')
      idx += 1
  fig.savefig("/content/gan_images/%d.png" % epoch)
  plt.close()

# train Discriminator(with real imgs and fake imgs)
# Main training loop
for epoch in range(epochs):
  ###########################
  ### Train discriminator ###
  ###########################
  # Select a random batch of images
  idx = np.random.randint(0, x_train.shape[0], batch_size)
  real_imgs = x_train[idx]
  # Generate fake images
  noise = np.random.randn(batch_size, latent_dim)  #generator to generate fake imgs
  fake_imgs = generator.predict(noise)
  # Train the discriminator
  # both loss and accuracy are returned
  d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)  #belong to positive class(real imgs)
  d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)  #fake imgs
  d_loss = 0.5 * (d_loss_real + d_loss_fake)
  d_acc  = 0.5 * (d_acc_real + d_acc_fake)
  #######################
  ### Train generator ###
  #######################
  noise = np.random.randn(batch_size, latent_dim)
  g_loss = combined_model_gen.train_on_batch(noise, ones)
  #trying to fool the discriminator that generate imgs are real that's why we are providing label as 1
  # do it again!
  noise = np.random.randn(batch_size, latent_dim)
  g_loss = combined_model_gen.train_on_batch(noise, ones)
  # Save the losses
  d_losses.append(d_loss)  #save the loss at each epoch
  g_losses.append(g_loss)
  if epoch % 100 == 0:
    print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f}, d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")
    k.append(d_acc)
  if epoch % sample_period == 0:
    sample_images(epoch)

print(sum(k)/len(k))

import os

print(os.path)
