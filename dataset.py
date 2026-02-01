import tensorflow as tf
import tensorflow_datasets as tfds

# Trimmed down to 10 chosen UAE-related dishes
SELECTED_CLASSES = [
    'chicken_curry', 'grilled_salmon', 'pizza', 'samosa', 'garlic_bread',
    'steak', 'french_fries', 'hamburger', 'spaghetti_bolognese', 'falafel'
]

IMG_SIZE = (192, 192)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Augmentation
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

def _prep(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    # One-hot encode labels for MixUp
    label = tf.one_hot(label, len(SELECTED_CLASSES))
    return image, label

def _augment(image, label):
    return data_augmentation(image, training=True), label

def mixup(ds_one, ds_two, alpha=0.2):
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    
    batch_size = tf.shape(images_one)[0]
    # generate random mixing number
    gamma_1 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma_2 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    l = gamma_1 / (gamma_1 + gamma_2)
    
    l_img = tf.reshape(l, [batch_size, 1, 1, 1])
    l_lbl = tf.reshape(l, [batch_size, 1])
    
    images = l_img * images_one + (1 - l_img) * images_two
    labels = l_lbl * labels_one + (1 - l_lbl) * labels_two
    
    return images, labels

def get_dataset(
    selected_classes=None,
    batch_size=BATCH_SIZE,
    shuffle_buffer=1000,
    train_split='train',
    val_split='validation',
):
    if selected_classes is None:
        selected_classes = SELECTED_CLASSES

    (ds_train, ds_val), ds_info = tfds.load(
        'food101',
        split=[train_split, val_split],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    label_names = ds_info.features['label'].names
    selected_class_ids = [label_names.index(name) for name in selected_classes]
    ids_tensor = tf.constant(selected_class_ids, dtype=tf.int64)

    lookup_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=ids_tensor,
            values=tf.range(len(selected_class_ids), dtype=tf.int64)
        ),
        default_value=-1
    )

    def _keep(x, y):
        return tf.reduce_any(tf.equal(y, ids_tensor))

    def _map_and_prep(x, y):
        y_new = lookup_table.lookup(y)
        return _prep(x, y_new)

    train = (ds_train
             .filter(_keep)
             .map(_map_and_prep, num_parallel_calls=AUTOTUNE, deterministic=False)
             .cache()
             .shuffle(shuffle_buffer)
             .map(_augment, num_parallel_calls=AUTOTUNE, deterministic=False)
             # Drop remainder to ensure all batches are exactly 32
             .batch(batch_size, drop_remainder=True) 
             .prefetch(AUTOTUNE))

    train_mu = train.shuffle(1000)
    
    train_mixed = tf.data.Dataset.zip((train, train_mu))
    train_mixed = train_mixed.map(mixup, num_parallel_calls=AUTOTUNE)

    val = (ds_val
           .filter(_keep)
           .map(_map_and_prep, num_parallel_calls=AUTOTUNE)
           .cache()
           .batch(batch_size)
           .prefetch(AUTOTUNE))

    return train_mixed, val, list(range(len(selected_classes))), selected_classes