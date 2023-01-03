import cv2
import numpy as np
import skimage.exposure
import material_changer


def color_change(image_path, mask, desired_color=(180, 128, 200)):
    # specify desired bgr color for new face and make into array
    desired_color = np.asarray(desired_color, dtype=np.float64)

    # create swatch
    swatch = np.full((200,200,3), desired_color, dtype=np.uint8)

    # read image
    img = cv2.imread(image_path)

    # read face mask as grayscale and threshold to binary
    # uint_img = np.array(mask * 255).astype('uint8')
    # facemask = cv2.threshold(uint_img, 100, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('Mask', facemask); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    # compute difference colors and make into an image the same size as input
    # diff_color = desired_color - ave_color
    # diff_color = np.full_like(img, diff_color, dtype=np.uint8)
    diff_color = np.full_like(img, desired_color, dtype=np.uint8)

    # shift input image color
    # cv2.add clips automatically
    new_img = cv2.add(img, diff_color)
    # cv2.imshow('Colored Img', new_img); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    # antialias mask, convert to float in range 0 to 1 and make 3-channels
    # facemask = cv2.GaussianBlur(facemask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
    # facemask = skimage.exposure.rescale_intensity(facemask, in_range=(100,150), out_range=(0,1)).astype(np.float32)
    # facemask = cv2.merge([facemask,facemask,facemask])
    facemask = material_changer.post_process_mask(mask)
    cv2.imshow('Antialiased Mask', facemask); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    # combine img and new_img using mask
    # cropped_result = (img * (1 - facemask))
    # cropped_result = cropped_result.clip(0, 255).astype(np.uint8)
    cropped_result = material_changer.crop_image(img, facemask)
    cv2.imshow('Cropped Result', cropped_result); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    colored_facemask = (new_img * facemask)
    # colored_facemask = cv2.cvtColor(colored_facemask, cv2.COLOR_RGB2RGBA)
    colored_facemask = colored_facemask.clip(0, 255).astype(np.uint8)
    # print(colored_facemask.shape)
    cv2.imshow('Colored Mask', colored_facemask); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
    print(cropped_result.dtype)
    print(colored_facemask.dtype)
    result = material_changer.blend_transparent(cropped_result, colored_facemask)

    return swatch, result
