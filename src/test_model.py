import os
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
import skimage.exposure
from skimage.transform import resize
import tensorflow_hub as hub
import cv2
from color_change import color_change
from style_transfer_model import StyleTransferModel
matplotlib.use('TkAgg')


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        # np.set_printoptions(threshold=10_000_000, linewidth=224)
        # print("Title: " + title[i] + ", Tensor: " + str(display_list[i]))
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(sample_image, model):
    # display([sample_image, np.where(model.predict(sample_image[tf.newaxis, ...])[0] > 0.5, 1, 0)])
    # display([sample_image, model.predict(sample_image[tf.newaxis, ...])[0]])
    result = model.predict(sample_image[tf.newaxis, ...])[0]
    display([sample_image, result])
    return result


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


if __name__ == "__main__":
    # load label map

    # load model
    # model = tf.keras.models.load_model('xnet45.h5', compile=False)
    # model = tf.keras.models.load_model('wnet100.h5', compile=False)
    # model = tf.keras.models.load_model('unet500.h5', compile=False)
    # model = tf.keras.models.load_model('./qnet/qnet298.h5', compile=False)
    # model = tf.keras.models.load_model('./qnet/qnet386.h5', compile=False)
    # model = tf.keras.models.load_model('./qnet/qnet459.h5', compile=False)
    model = tf.keras.models.load_model('./pnet/pnet26.h5', compile=False)
    # hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # load image path
    while(True):
        # try:
            img_path = input("Give me a path to an image (Type '' to leave): ")
            if img_path != '':
                material_change_mode = input("Do you want to change color or texture (Type 'texture' or 'color'): ")
                if material_change_mode.lower() == 'texture':
                    style_path = input("Give me a path to a texture/style: ")
                    original_image = PIL.Image.open(img_path)
                    original_array = np.array(original_image)
                    resized_image = original_image.resize((224, 224))
                    resized_array = np.array(resized_image) / 255.
                    resized_array = resized_array.reshape((resized_array.shape[0], resized_array.shape[1], 3))

                    pred = model.predict(resized_array[tf.newaxis, ...])[0]
                    resized_pred1 = resize(pred, (original_array.shape[0], original_array.shape[1], 1))
                    display([original_image, resized_pred1])

                    style_model = StyleTransferModel([img_path], resized_pred1 ,[style_path])
                    styles = style_model.run()
                    im = PIL.Image.fromarray(styles)
                    plt.imshow(im)
                    plt.axis("off")
                    plt.show()

                    # style = tf.constant(load_img(style_path))
                    # picture = tf.constant(load_img(img_path))
                    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                    # stylized_image = hub_model(picture, style)[0]
                    # final_result = tensor_to_image(stylized_image)
                    # plt.imshow(final_result)
                    # plt.axis('off')
                    # plt.show()
                else:
                    color = input("Give me a color (b, g, r): ")
                    original_image = PIL.Image.open(img_path)
                    original_array = np.array(original_image)
                    resized_image = original_image.resize((224, 224))
                    resized_array = np.array(resized_image) / 255.
                    resized_array = resized_array.reshape((resized_array.shape[0], resized_array.shape[1], 3))

                    pred = model.predict(resized_array[tf.newaxis, ...])[0]
                    resized_pred1 = resize(pred, (original_array.shape[0], original_array.shape[1], 1))
                    display([original_image, resized_pred1])

                    new_color, result = color_change(img_path, resized_pred1, desired_color=eval(color))
                    plt.imshow(result)
                    plt.axis("off")
                    plt.show()
            else:
                break
        # except Exception as e:
        #     print("That is not a valid input... " + str(e))
