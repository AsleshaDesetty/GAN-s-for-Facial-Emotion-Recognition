
#importing libraries
from sklearn.model_selection import train_test_split
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import cv2 as cv
import glob
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
import glob 
import imageio
import matplotlib.pyplot as plt 
import numpy as np
import os
import PIL
from tensorflow.keras import layers 
import time
import tensorflow as tf
from IPython import display

#mounting google drive
#variables
#loading images and their label
from google.colab import drive 
drive.mount('/content/drive')

x = []
y = []
count = 0
path = "/content/drive/MyDrive/own_dataset/anger"
img_files = os.listdir(path)
i = 0
for img in glob.glob("/content/drive/MyDrive/own_dataset/anger/*.jpg"):
    count = count + 1
    # loading images and their label
    img = cv.imread(path + '/' + img_files[i])
    x.append(img)
    y.append(0)
    i = i + 1

path = "/content/drive/MyDrive/own_dataset/contempt"
img_files = os.listdir(path)
i = 0
for img in glob.glob("/content/drive/MyDrive/own_dataset/contempt/*.jpg"):
    count = count + 1
    img = cv.imread(path + '/' + img_files[i])
    x.append(img)
    y.append(1)
    i = i + 1

path = "/content/drive/MyDrive/own_dataset/disgust"
img_files = os.listdir(path)
i = 0
for img in glob.glob("/content/drive/MyDrive/own_dataset/disgust/*.jpg"):
    count = count + 1
    img = cv.imread(path + '/' + img_files[i])
    x.append(img)
    y.append(2)
    i = i + 1

path = "/content/drive/MyDrive/own_dataset/fear"
img_files = os.listdir(path)
i = 0
for img in glob.glob("/content/drive/MyDrive/own_dataset/fear/*.jpg"):
    count = count + 1
    img = cv.imread(path + '/' + img_files[i])
    x.append(img)
    y.append(3)
    i = i + 1

path = "/content/drive/MyDrive/own_dataset/happy"
img_files = os.listdir(path)
i = 0
for img in glob.glob("/content/drive/MyDrive/own_dataset/happy/*.jpg"):
    count = count + 1
    img = cv.imread(path + '/' + img_files[i])
    x.append(img)
    y.append(4)
    i = i + 1

path = "/content/drive/MyDrive/own_dataset/sadness"
img_files = os.listdir(path)
i = 0
for img in glob.glob("/content/drive/MyDrive/own_dataset/sadness/*.jpg"):
    count = count + 1
    img = cv.imread(path + '/' + img_files[i])
    x.append(img)
    y.append(5)
    i = i + 1

path = "/content/drive/MyDrive/own_dataset/surprise"
img_files = os.listdir(path)
i = 0
for img in glob.glob("/content/drive/MyDrive/own_dataset/surprise/*.jpg"):
    count = count + 1
    img = cv.imread(path + '/' + img_files[i])
    x.append(img)
    y.append(6)
    i = i + 1

#printing number of total images
print("Total number of images available in the dataset:" + str(count))

#splitting the data into train, test, and validation data
# summarize the shape of the dataset
trainX, X_rem, trainy, y_rem = train_test_split(x, y, train_size=0.8)
validX, testX, validy, testy = train_test_split(X_rem, y_rem, test_size=0.5)
trainX = np.asarray(trainX)
testX = np.array(testX)
trainy = np.asarray(trainy)
testy = np.array(testy)
validX = np.array(validX)

from tensorflow.keras.utils import to_categorical

#Plotting the data
validy = np.array(validy)
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)
print('Validation', validX.shape, validy.shape)

trainy = to_categorical(trainy, 7)
testy = to_categorical(testy, 7)
validy = to_categorical(validy, 7)

from tensorflow.keras.datasets.mnist import load_data
from matplotlib import pyplot

for i in range(25):
    pyplot.subplot(5, 5, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(trainX[i], cmap='gray_r')

    pyplot.show()

#initializing the train_images, train labels
BUFFER_SIZE = 3000
BATCH_SIZE = 256

#shuffling the

 data
#generator definition
x = np.asarray(x)
y = np.asarray(x)
(train_images, train_labels) = (x, y)
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((16, 16, 256)))
    assert model.output_shape == (None, 16, 16, 256)  # Note: None is the batch size

    # upsample to 32*32
    # upsample to 64*64

    # calling the generator
    # defining the discriminator
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model


generator = make_generator_model()
noise = tf.random.normal([1, 100])
#latent space
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    # 2×2 stride to downsample
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))  # downsampling 2×2 stride to downsample
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(7, activation='softmax'))
    return model


# calling the discriminator
discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


# defining the discriminator loss
# defining the generator loss
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])
generator = make_generator_model()
decision = discriminator(generated_image)
print(decision)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Adding checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer, generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50


@tf.function
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# Training the dataset
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)

        if (epoch + 1)

 % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    display.clear_output(wait=True)


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


history = discriminator.fit(trainX, trainy, validation_data=(validX, validy), epochs=20)  # accuracy graph

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.xlabel('acc')
plt.ylabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# loss graph
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('LOSS')
plt.ylabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# saving file with .h5 extension
discriminator.save("final_model.h5")
# discriminator=tf.keras.models.load_model("/content/drive/MyDrive/final_model.h5")
performance = discriminator.evaluate(testX, testy, verbose=1)
print('Accuracy Test : {}'.format(performance[1]))

pred_labels = discriminator.predict(testX).argmax(axis=1)
true_labels = testy.argmax(axis=1)
confusion_mat = tf.math.confusion_matrix(labels=true_labels, predictions=pred_labels).numpy()
confusion_mat

import pandas as pd
import seaborn as sns

confusion_mat_norm = np.around(confusion_mat.astype('float') / np.atleast_2d(confusion_mat.sum(axis=1)).T, decimals=2)
classes = np.arange(0, 7).astype('str')
figure = plt.figure()

# plot the evolution of Loss and Accuracy on the train and validation sets
import matplotlib.pyplot as plt
confusion_mat_df = pd.DataFrame(confusion_mat_norm, index=classes, columns=classes)
sns.heatmap(confusion_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.subplot(1, 2, 2)

from tensorflow.keras.preprocessing import image
import PIL

# definition for testing
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()


def test_model(image_for_testing):
    test_image = image.load_img(image_for_testing, target_size=(48, 48), color_mode='grayscale')
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = discriminator.predict(test_image)
    result = np.round(result).astype(int)
    print(result)


Catagories = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

image_show = PIL.Image.open(image_for_testing)
plt.imshow(image_show)
#print(int(result[0][0]))
plt.title(Catagories[np.argmax(result[0])])
plt.axis("off")
plt.show()

# reading an image
import cv2

img = cv2.imread("/content/drive/MyDrive/own_dataset/happy/113.jpg")
img.shape
img = cv2.resize(img, (64, 64))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray.shape

# detection of face
faces = faceCascade.detectMultiScale(gray, 10, 28)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    face_roi = img[y:y + h, x:x + w]

#

 eyes detection runs for each face
    facess = faceCascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in facess:
        cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), 255, 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

final_image = cv2.resize(img, (64, 64))
final_image = np.expand_dims(final_image, axis=0)
final_image = final_image / 255.0

Predictions = discriminator.predict(final_image)

c = np.argmax(Predictions)

if c == 0:
    print("angry")
elif c == 1:
    print("contempt")
elif c == 2:
    print("disgust")
elif c == 3:
    print("fear")
elif c == 4:
    print("happy")
elif c == 5:
    print("sad")
elif c == 6:
    print("surprise")
