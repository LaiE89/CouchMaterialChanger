import cv2
import numpy as np
import skimage.exposure


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

    # print the alpha value on the image
    # cv2.putText(bg_img, f'Alpha: {round(alpha, 1)}', (50, 200),
    #             cv2.FONT_HERSHEY_PLAIN, 8, (200, 200, 200), 7)

    # resize the image before displaying
    cv2.imshow('Final Overlay', bg_img)
    cv2.waitKey(0)
    return cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)


def color_change(image_path, mask, desired_color=(180, 128, 200)):
    # specify desired bgr color for new face and make into array
    desired_color = np.asarray(desired_color, dtype=np.float64)

    # create swatch
    swatch = np.full((200,200,3), desired_color, dtype=np.uint8)

    # read image
    img = cv2.imread(image_path)

    # read face mask as grayscale and threshold to binary
    uint_img = np.array(mask * 255).astype('uint8')
    # facemask = cv2.cvtColor(uint_img, cv2.COLOR_BGR2GRAY)
    facemask = cv2.threshold(uint_img, 128, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('facemask', facemask); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
    # facemask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # get average bgr color of face
    # print(facemask.shape)
    # print(img.shape)
    # ave_color = cv2.mean(img, mask=facemask)[:3]
    # print(ave_color)

    # compute difference colors and make into an image the same size as input
    # diff_color = desired_color - ave_color
    # diff_color = np.full_like(img, diff_color, dtype=np.uint8)
    diff_color = np.full_like(img, desired_color, dtype=np.uint8)

    # shift input image color
    # cv2.add clips automatically
    new_img = cv2.add(img, diff_color)
    cv2.imshow('New Img', new_img); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    # antialias mask, convert to float in range 0 to 1 and make 3-channels
    facemask = cv2.GaussianBlur(facemask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
    facemask = skimage.exposure.rescale_intensity(facemask, in_range=(100,150), out_range=(0,1)).astype(np.float32)
    facemask = cv2.merge([facemask,facemask,facemask])
    cv2.imshow('facemask', facemask); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    # combine img and new_img using mask
    cv2.imshow('original image', img); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
    cropped_result = (img * (1 - facemask))
    #cropped_result = cv2.cvtColor(cropped_result, cv2.COLOR_RGB2RGBA)
    cropped_result = cropped_result.clip(0, 255).astype(np.uint8)
    print(cropped_result.shape)
    cv2.imshow('cropped result', cropped_result); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    colored_facemask = (new_img * facemask)
    #colored_facemask = cv2.cvtColor(colored_facemask, cv2.COLOR_RGB2RGBA)
    colored_facemask = colored_facemask.clip(0, 255).astype(np.uint8)
    print(colored_facemask.shape)
    cv2.imshow('colored facemask', colored_facemask); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    result = blend_transparent(cropped_result, colored_facemask)

    return swatch, result
