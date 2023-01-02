import tensorflow as tf
from model import get_lr_metric

#model = tf.keras.models.load_model('xnet45.h5', custom_objects={'IoU': IoU})
#model = tf.keras.models.load_model('vnet98.h5', custom_objects=None)
#model = tf.keras.models.load_model('unet498.h5', custom_objects=None)
#model = tf.keras.models.load_model('./tnet/tnet98.h5', custom_objects=None)
#model = tf.keras.models.load_model('./rnet/rnet565.h5', custom_objects={"lr": get_lr_metric})
#model = tf.keras.models.load_model('./qnet/qnet386.h5', custom_objects={"IoU": tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5), "lr": get_lr_metric})
#model = tf.keras.models.load_model('./qnet/qnet459.h5', custom_objects={"IoU": tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5), "lr": get_lr_metric})
model = tf.keras.models.load_model('./pnet/pnet26.h5', custom_objects={"IoU": tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5), "lr": get_lr_metric})
model.save_weights('./checkpoints/my_checkpoint')