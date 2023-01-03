import cv2
import numpy as np
import skimage.exposure
from skimage import img_as_ubyte


def blend_transparent(background, overlay_image):
    h, w = overlay_image.shape[:2]

    # Create a new np array
    shapes = np.zeros_like(background, np.uint8)

    # Put the overlay at the bottom-right corner
    shapes[background.shape[0] - h:, background.shape[1] - w:] = overlay_image

    # Change this into bool to use it as mask
    mask = shapes.astype(bool)

    # We'll create a loop to change the alpha
    # value i.e transparency of the overlay
    alpha = 1
    # for alpha in np.arange(0, 1.1, 0.1)[::-1]:
    # Create a copy of the image to work with
    bg_img = background.copy()
    # Create the overlay
    bg_img[mask] = cv2.addWeighted(bg_img, 1 - alpha, shapes,
                                   alpha, 0)[mask]

    return cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)


def post_process_mask(mask):
    uint_img = np.array(mask * 255).astype('uint8')
    binary_mask = cv2.threshold(uint_img, 100, 255, cv2.THRESH_BINARY)[1]

    binary_mask = cv2.GaussianBlur(binary_mask, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
    binary_mask = skimage.exposure.rescale_intensity(binary_mask, in_range=(100, 150), out_range=(0, 1)).astype(
        np.float32)
    binary_mask = cv2.merge([binary_mask, binary_mask, binary_mask])
    return binary_mask


def crop_image(img, mask):
    cropped_result = (img * (1 - mask))
    cropped_result = cropped_result.clip(0, 255).astype(np.uint8)
    return cropped_result


def crop_mask_from_image(img, mask):
    cropped_result = (img * mask)
    cropped_result = cropped_result.clip(0, 255)
    return cropped_result


def convert_float64_to_uint8(array):
    # info = np.finfo(array.dtype)
    # data = array.astype(np.float64) / info.max  # normalize the data to 0 - 1
    # data = 255 * data  # Now scale by 255
    # return data.astype(np.uint8)
    return img_as_ubyte(array)
