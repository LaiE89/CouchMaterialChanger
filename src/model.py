import tensorflow as tf
from model_parts import upsample
from matplotlib import pyplot as plt, gridspec
import numpy as np


class Unet:
    output_channels = 1

    def __init__(self):
        # tf.config.experimental_run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()
        base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        down_stack.trainable = False

        up_stack = [
            upsample(1024, 3, norm_type='batchnorm', apply_dropout=True, kernel_regularizer="l2"),  # 4x4 -> 8x8
            upsample(512, 3, norm_type='batchnorm', apply_dropout=True, kernel_regularizer="l2"),  # 8x8 -> 16x16
            upsample(256, 3, norm_type='batchnorm', apply_dropout=True, kernel_regularizer="l2"),  # 16x16 -> 32x32
            upsample(128, 3, norm_type='batchnorm', apply_dropout=False, kernel_regularizer="l2"),  # 32x32 -> 64x64
            upsample(64, 3, norm_type='batchnorm', apply_dropout=True, kernel_regularizer="l2"),  # 32x32 -> 64x64
        ]
        inputs = tf.keras.layers.Input(shape=[224, 224, 3])

        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
          x = up(x)
          concat = tf.keras.layers.Concatenate()
          x = concat([x, skip])

        last = tf.keras.layers.Conv2DTranspose(
            self.output_channels,
            kernel_size=3, strides=2,
            padding='same', activation='sigmoid')  # (bs, 256, 256, 3)

        x = last(x)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            #0.001,
            0.0000001,
            decay_steps=978,
            decay_rate=1,
            staircase=True,
        )

        #optimizer = tf.keras.optimizers.Adam(lr_schedule)
        optimizer = tf.keras.optimizers.Nadam(1e-4)
        lr_metric = get_lr_metric(optimizer)
        metrics = [tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5), lr_metric]

        self.model = tf.keras.Model(inputs=inputs, outputs=x)

        self.model.compile(
                  optimizer=optimizer,
                  #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  # run_eagerly=True,
                  metrics=metrics)

    def get_model(self):
        return self.model


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)  # I use ._decayed_lr method instead of .lr

    return lr


def visualize(npa, npa2):
    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                     subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if (i == 1):
                ax.matshow(npa[j], cmap='Greens')
            else:
                ax.matshow(npa2[j], cmap='Blues')

            # ax.axis('off')
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


unet = Unet().get_model()

unet.summary()
