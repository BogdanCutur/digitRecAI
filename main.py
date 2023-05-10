import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pygame
from skimage.transform import resize

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train_flat = np.reshape(x_train, (x_train.shape[0], -1))
x_test_flat = np.reshape(x_test, (x_test.shape[0], -1))

noise_factor = .08
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# plt.subplot(1, 2, 1)
# plt.imshow(x_train[5], cmap='gray')
# plt.title('Original Image')
#
# # Plot the first noisy image in the training set
# plt.subplot(1, 2, 2)
# plt.imshow(x_train_noisy[5], cmap='gray')
# plt.title('Noisy Image')
#
# plt.show()

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
#
# model.fit(x_train_flat, y_train, epochs=5, validation_data=(x_test_flat, y_test))
#
# test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=2)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

#
# model.fit(x_train_noisy, y_train, epochs=5, validation_data=(x_test_noisy, y_test))
#
# model.save_weights('autoencoder_weights.h5')

model.load_weights('autoencoder_weights.h5')

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print("Loss: ", test_loss)
print("Accuracy: ", test_acc)

pygame.init()
window_size = (280, 280)
background_color = (0, 0, 0)
brush_color = (255, 255, 255)
brush_size = 10
# Create the window
screen = pygame.display.set_mode(window_size)
# Set the window title
pygame.display.set_caption('Handwritten Digit Recognition')
# Initialize the drawing surface
drawing_surface = pygame.Surface((280, 280))
# Set the drawing surface background color
drawing_surface.fill(background_color)
running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pygame.draw.circle(drawing_surface, brush_color, event.pos, brush_size)
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0] == 1:
                pygame.draw.circle(drawing_surface, brush_color, event.pos, brush_size)
        elif event.type == pygame.KEYDOWN:
            # If the user presses the space bar, recognize the digit
            if event.key == pygame.K_SPACE:
                drawing_surface.fill(background_color)
            if event.key == pygame.K_p:
                digit_image = pygame.surfarray.array3d(drawing_surface)
                digit_image = np.flip(digit_image, 0)
                digit_image = np.rot90(digit_image, -1)

                digit_image = resize(digit_image, (28, 28))

                digit_image = np.dot(digit_image[..., :3], [1, 1, 1])

                digit_image = (digit_image - digit_image.min()) / (
                            digit_image.max() - digit_image.min())


                # plt.subplot(1, 2, 1)
                # plt.imshow(digit_image, cmap='gray')
                # plt.title('Original Image')
                #
                # plt.show()

                digit_image = digit_image.reshape((1, 28, 28))

                print(digit_image)
                # print(digit_image.shape)
                # print(x_train[0].shape)

                prediction = model.predict(digit_image)

                # Get the predicted digit
                predicted_digit = np.argmax(prediction)

                # Print the predicted digit to the console
                print("Prediction is: ", predicted_digit)

        screen.blit(drawing_surface, (0, 0))

        # Update the display
        pygame.display.flip()

pygame.quit()
