from abc import ABC
from collections.abc import Iterable
import random
import numpy as np
import skimage.io as io
import cv2
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt, gridspec


def get_class_name(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return None


def get_image(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name'])/255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img


def get_normal_mask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = get_class_name(anns[a]['category_id'], cats)
        pixel_value = classes.index(className) + 1
        new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def get_binary_mask(imageObj, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)

        # Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


class DataGenerator(Iterable, ABC):
    augGeneratorArgs = dict(featurewise_center=False,
                            samplewise_center=False,
                            rotation_range=3,
                            width_shift_range=0.01,
                            height_shift_range=0.01,
                            zoom_range=[1, 1.25],
                            shear_range=0.01,
                            horizontal_flip=True,
                            vertical_flip=False,
                            fill_mode="constant",
                            data_format = 'channels_last')

    images = []

    def __init__(self, images, classes, coco, folder, input_image_size=(224,224), batch_size=4, mode='train', mask_type='binary'):
        self.images = self.generate(images, classes, coco, folder, input_image_size, batch_size, mode, mask_type)

    @staticmethod
    def generate(images, classes, coco, folder, input_image_size=(224,224), batch_size=4, mode='train', mask_type='binary'):

        img_folder = '{}/images/{}'.format(folder, mode)
        dataset_size = len(images)
        catIds = coco.getCatIds(catNms=classes)

        c = 0
        while (True):
            img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
            mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

            for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
                imageObj = images[i]

                ### Retrieve Image ###
                train_img = get_image(imageObj, img_folder, input_image_size)

                ### Create Mask ###
                if mask_type == "binary":
                    train_mask = get_binary_mask(imageObj, coco, catIds, input_image_size)

                elif mask_type == "normal":
                    train_mask = get_normal_mask(imageObj, classes, coco, catIds, input_image_size)

                    # Add to respective batch sized arrays
                img[i - c] = train_img
                mask[i - c] = train_mask

            c += batch_size
            if (c + batch_size >= dataset_size):
                c = 0
                random.shuffle(images)
            mask = np.where(mask > 0.5, 1, 0)
            yield img, mask

    def visualize(self):
        img, mask = next(self)

        fig = plt.figure(figsize=(20, 10))
        outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

        for i in range(2):
            innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                         subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

            for j in range(4):
                ax = plt.Subplot(fig, innerGrid[j])
                if (i == 1):
                    ax.imshow(img[j])
                else:
                    ax.imshow(mask[j][:, :, 0])

                ax.axis('off')
                fig.add_subplot(ax)
        plt.show()

    def augment(self, seed=None):
        # Initialize the image data generator with args provided
        image_gen = ImageDataGenerator(**self.augGeneratorArgs)

        np.random.seed(seed if seed is not None else np.random.choice(range(9999)))

        for img, mask in self:
            seed = np.random.choice(range(9999))
            # keep the seeds syncronized otherwise the augmentation of the images
            # will end up different from the augmentation of the masks
            g_x = image_gen.flow(255 * img, batch_size=img.shape[0], seed=seed, shuffle=True)
            g_y = image_gen.flow(mask, batch_size=mask.shape[0], seed=seed, shuffle=True)

            img_aug = next(g_x) / 255.0

            mask_aug = next(g_y)
            yield img_aug, mask_aug

    def __iter__(self):
        # super().__iter__()
        return self.images.__iter__()

    def __next__(self):
        return self.images.__next__()

