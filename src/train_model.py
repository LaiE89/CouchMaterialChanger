import random
import tensorflow as tf
from IPython.display import clear_output
import data_generator
from pycocotools.coco import COCO
from itertools import islice
from model import unet
from matplotlib import pyplot as plt, gridspec
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



def filter_dataset(folder, classes=None, mode='train'):
    # initialize COCO api for instance annotations
    annFile = '{}/annotations/instances_{}2017.json'.format(folder, mode)
    coco = COCO(annFile)
    lower_bound = 5
    upper_bound = 95
    images = []

    if classes != None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            # images += coco.loadImgs(imgIds)
            for id in imgIds:
                annIds = coco.getAnnIds(imgIds=id)
                anns = coco.loadAnns(annIds)
                # couch_count = np.sum([ann['category_id'] == 63 for ann in anns])
                # if couch_count == 1:
                anns = list(filter(lambda ann: ann['category_id'] == 63, anns))

                ann1 = anns[0]
                img1 = coco.loadImgs(ann1['image_id'])[0]
                img1_area = img1['height'] * img1['width']
                img1_ratio = ann1['area'] / img1_area * 100
                # print(img1_ratio)
                if lower_bound < img1_ratio < upper_bound:
                    images += coco.loadImgs(id)
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)

    return unique_images, len(unique_images), coco


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions():
    #display([sample_image, sample_mask, create_mask(unet.predict(sample_image[tf.newaxis, ...]))])
    #display([sample_image, sample_mask, unet.predict(sample_image[tf.newaxis, ...])[0]])
    return


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    # show_predictions()
    cur_image = list(islice(train_gen, 1))[0]
    prediction = unet.predict(cur_image[0][0][tf.newaxis, ...])[0]
    # get_mask_details(prediction)
    display([cur_image[0][0], cur_image[1][0], prediction])
    #display([cur_image[0][0], cur_image[1][0], create_mask(unet.predict(cur_image[0][0][tf.newaxis, ...]))])
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


def display_training(model):
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    plt.figure()
    plt.plot(model.epoch, loss, 'r', label='Training loss')
    plt.plot(model.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


class SaveModel(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save('./pnet/pnet{}.h5'.format(epoch + 26 + 1), overwrite=True,)


def train():
    EPOCHS = 250
    VAL_SUBSPLITS = 2
    VALIDATION_STEPS = dataset_size // batch_size // VAL_SUBSPLITS
    STEPS_PER_EPOCH = train_dataset_size // batch_size
    unet.load_weights('./checkpoints/my_checkpoint')

    history = unet.fit(train_gen, epochs=EPOCHS,
                       steps_per_epoch=STEPS_PER_EPOCH,
                       validation_steps=VALIDATION_STEPS,
                       validation_data=val_gen,
                       callbacks=[SaveModel(),
                                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
                                  EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
                                  ],
                       #callbacks=[DisplayCallback()],
                       )

    display_training(history)
    #unet.save("CouchSegmentation.h5")  # we can save the model and reload it at anytime in the future


def visualize(gen):
    img, mask = next(gen)

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


def get_mask_details(mask):
    np.set_printoptions(threshold=10_000_000, linewidth=np.inf)
    for i in range(0, len(mask)):
        # print("column " + str(i) + ": " + str(sample_mask[i]))
        print(" ".join([str(v) for v in mask[i]]))
    print(mask.shape)
    plt.matshow(mask, cmap='Greens')
    plt.show()


if __name__ == "__main__":
    folder = './COCOdataset2017'
    classes = ['couch']
    images, dataset_size, coco = filter_dataset(folder, classes, 'val')
    train_images, train_dataset_size, train_coco = filter_dataset(folder, classes, 'train')
    print("Validation dataset size: " + str(dataset_size))

    batch_size = 4
    input_image_size = (224, 224)
    mask_type = 'binary'

    val_gen = data_generator.DataGenerator(images, classes, coco, folder, input_image_size, batch_size, 'val', mask_type)
    train_gen = data_generator.DataGenerator(train_images, classes, train_coco, folder, input_image_size, batch_size, 'train', mask_type).augment()

    # visualize(val_gen)
    # visualize(train_gen)

    # for images, masks in train_gen:
    #     sample_image, sample_mask = images[0], masks[0]
    #     break

    # for i in range(2):
    #     next_image = val_gen.__next__()
    #     sample_image, sample_mask = next_image[0][0], next_image[1][0]
    #     show_predictions()

    train()

