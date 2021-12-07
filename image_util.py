from tensorflow.keras.preprocessing import image
import numpy as np


def load_image(image_path):
    """

    :param image_path:
    :return: shape is [1, 3, 224, 224], value is [0, 1]
    """
    target_image = image.load_img(image_path, target_size=(224, 224))
    target_image = image.img_to_array(target_image) / 255.
    target_image = np.expand_dims(target_image.transpose(2, 0, 1), axis=0)
    return target_image


def save_image(image_path, img):
    """

    :param image_path:
    :param img: shape is [224, 224, 3], value is [0, 1]
    :return:
    """
    image.save_img(image_path, img)


def squeeze_transpose(img):
    """

    :param img: shape is [1, 3, None, None]
    :return: [None, None, 3]
    """
    return np.squeeze(img).transpose(1, 2, 0)
