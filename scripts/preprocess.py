import pandas as pd
from numpy import copy
import json
import os
import glob
import string
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from numpy import random

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset

import imgaug.augmenters as iaa
import imageio


def meta_collect(ann_path: str, result_file: str,
                 ready: int = 1, sep: str = '\t') -> None:
    '''collect metadata for all images to "result_file"
    from json files in "ann_path" (execution time: about 5 mins)'''

    if not ready:
        return

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(sep.join(['width', 'height', 'description',
                          'isModerated', 'moderatedBy', 'predicted']) + '\n')

        for file in tqdm(glob.glob(os.path.join(ann_path, '*.json'))):

            with open(file, encoding='utf-8') as js:
                tmp = json.load(js)

            try:
                f.write(sep.join([tmp['name'], str(tmp['size']['width']), str(tmp['size']['height']),
                                  tmp['description'], str(tmp['moderation']['isModerated']),
                                  tmp['moderation']['moderatedBy'], str(tmp['moderation']['predicted'])]) + '\n')
            except Exception:
                print(tmp['description'])


def make_augments(df: pd.DataFrame, num: int = 1, ready: int = 1) -> str:
    paths = df.index.to_series().apply(lambda x: os.path.join(img_path, x) + '.jpg')

    aug_1 = os.path.join(img_path, 'aug_1')
    if not ready:
        return aug_1

    if not os.path.exists(aug_1):
        os.mkdir(aug_1)

    seq = iaa.Sequential(
        [

            iaa.Sometimes(0.4, iaa.GaussianBlur(3.0)),

            iaa.Sometimes(0.3, iaa.AveragePooling(2)),
            iaa.Sometimes(0.2, iaa.Emboss(alpha=(0.0, 1.0), strength=(0.75, 1.25))),
            iaa.Sometimes(0.4, iaa.GammaContrast((0.5, 1.0))),
            iaa.Invert(0.05, per_channel=True),
            iaa.Sometimes(0.3, iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))),

            iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),

            iaa.PerspectiveTransform(scale=(0.02, 0.05)),

            iaa.Sometimes(0.2, iaa.SaltAndPepper(0.05)),
        ],
        random_order=True
    )

    if num == 1:
        paths = [paths[i:i + 1000] for i in range(0, len(paths), 1000)]
        for path in tqdm(paths):
            img = [imageio.imread(i) for i in path]
            ls = seq(images=img)

            for i in range(len(path)):
                _, name = os.path.split(path[i])
                name = os.path.join(aug_1, f'aug_' + name)
                cv2.imwrite(name, ls[i])

    for path in tqdm(paths):
        img = imageio.imread(path)
        image = [copy(img) for _ in range(num)]

        ls = seq(images=image)

        for i in range(num):
            _, name = os.path.split(path)
            name = os.path.join(aug_1, f'{i}_aug_' + name)
            cv2.imwrite(name, ls[i])

    return aug_1


class PreprocessFrame(pd.DataFrame):

    def __init__(self, metadata: str = 'data/metadata.tsv', img_height: int = 100,
                 rem_str: str = '', img_width: int = 600,
                 subs_str: str = '', *args, **kwargs) -> None:

        super().__init__(self.__initial_start(metadata), *args, **kwargs)

        self.__rework(rem_str, subs_str)
        self = self[(self.width <= img_width) & (self.height <= img_height)]

    def __initial_start(self, x):
        '''Function for openning different extensions table files'''

        if isinstance(x, (list, tuple)):
            return {'x': x[0], 'y': x[1]}

        elif isinstance(x, pd.DataFrame):
            return x

        else:
            _, file_extension = os.path.splitext(x)
            try:
                if file_extension == '.csv':
                    return pd.read_csv(x)
                elif file_extension == '.tsv':
                    return pd.read_csv(x, sep='\t')
                else:
                    return pd.read_exel(x)
            except Exception:
                print('Cant open metadata file')
                return 1

    def __rework(self, rem_str: str, subs_str: str) -> None:
        '''/df'''

        if all(self.columns.isin(['predicted', 'isModerated'])):
            self = self.drop(['predicted', 'isModerated'], axis=1)

        self['description'] = self.description.str.replace('o', 'о').str.replace('H', 'Н')
        self['description'] = self.description.str.replace('–', '-').str.replace('—', '-').str.replace('…', '...')

        for r, s in zip(list(rem_str), list(subs_str)):
            self.description = self.description.str.replace(r, s)

        alphabet_lower = [chr(ord("а") + i) for i in range(32)] + [chr(ord("а") + 33)]  # Last is "ё"
        alphabet_upper = [chr(ord("А") + i) for i in range(32)]
        punctuation = list(string.punctuation)

        alphabet = set(alphabet_lower + alphabet_upper + punctuation)

        # Creating alphabet from dataset
        counts = self.counts_to_df()
        counts_dict = counts.set_index('symbols')['counts'].to_dict()

        # difference between dataset and reference alphabet
        bad_symbols = set(counts_dict) - alphabet
        self = self.drop(counts[counts.symbols.isin(bad_symbols)].index.drop_duplicates(), axis=0)

    def counts_to_df(self, column: str = 'description') -> pd.DataFrame:
        '''Return dataframe with symbols counts in "column"'''

        counts = pd.DataFrame(self[column].str.split('').explode())
        counts = counts.join(counts[column].value_counts(), on=column, rsuffix='1')
        counts.rename(columns={column: 'symbols',
                               column + '1': 'counts', },
                      inplace=True)
        counts = counts[~counts.isin(['', ' '])].dropna()

        return counts

    def train_test_val_split(self, test_size: float, val_size: float,
                             column: str = 'description', *args, **kwargs):
        ''' '''

        counts = self.counts_to_df(column)
        counts.counts = 1
        splitter = counts.reset_index().drop_duplicates().pivot(index='index', columns='symbols').fillna(0)

        train, test, _, ls = train_test_split(self, splitter,
                                              test_size=(test_size + val_size),
                                              *args, **kwargs)

        test, val, _, _ = train_test_split(test, ls, shuffle=True,
                                           test_size=(val_size / (test_size + val_size)),
                                           random_state=12)

        return PreprocessFrame(train), PreprocessFrame(test), PreprocessFrame(val)


class Dataset(PrefetchDataset):

    def __init__(self, df: PreprocessFrame, test_size: float, val_size: float,
                 batch_size: int = 16, img_height=100, img_width=600,
                 max_length=None, shuffle_buffer: int = 1024,
                 prefetch: int = tf.data.experimental.AUTOTUNE) -> None:

        self.df = df
        # Constants
        self.img_height = img_height if img_height else self.df.height.max()
        self.img_width = img_width if img_width else self.df.width.max()
        self.max_length = max_length if max_length else self.df.description.str.len().max()

        self.get_dataset(batch_size=batch_size, shuffle_buffer=shuffle_buffer,
                         prefetch=prefetch, test_size=test_size, val_size=val_size)

    def get_dataset(self, batch_size: int, test_size: float,
                    shuffle_buffer: int, prefetch: int,
                    val_size: float, *args, **kwargs) -> tf.data.Dataset:
        """Function for creating tf dataset"""

        # Creating mappers

        # Mapping characters to integers
        counts = self.df.counts_to_df()
        counts = counts.symbols.unique().tolist() + [' ', '#']

        vocab = pd.Series(counts).apply(lambda x: x.encode('utf8'))

        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=vocab,
            mask_token=None,
        )

        # Mapping integers back to original characters
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(),
            mask_token=None, invert=True,
        )

        self.blank_index = self.char_to_num(tf.strings.unicode_split('#', input_encoding="UTF-8")).numpy()[0]

        train, test, val = df.train_test_val_split(test_size=test_size,
                                                   val_size=val_size,
                                                   *args, **kwargs)
        for tmp in [train, test, val]:
            ind = tmp.index.tolist()
            random.shuffle(ind)
            tmp = tmp.loc[ind]
            tmp = tf.data.Dataset.from_tensor_slices(
                (tmp.index.to_series().apply(lambda x: os.path.join(img_path, x) + '.jpg').tolist(),
                 tmp.description.tolist())
            )

            tmp = (
                tmp.map(
                    self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
                .batch(batch_size)
                .prefetch(prefetch)
            )

            yield super().__init__(tmp, prefetch)

    def encode_single_sample(self, img_path: str, label: str) -> dict:
        """Function for processing one image from tf dataset"""

        # 1. Read
        img = tf.io.read_file(img_path)

        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)

        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)

        # 4. Resize to the desired size
        img = 1 - img
        img = tf.image.resize_with_crop_or_pad(img, self.img_height, self.img_width)
        img = 0.5 - img

        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])

        # 6. Map the characters in label to numbers
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        label = tf.pad(label, [[0, self.max_length - len(label)]], constant_values=self.blank_index)

        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}


if __name__ == '__main__':
    ann_path = os.path.join('HKR_Dataset_Words_Public', 'ann')
    img_path = os.path.join('HKR_Dataset_Words_Public', 'img')

    meta_collect(ann_path, 'data/metadata.tsv', 0)

    df = PreprocessFrame(metadata='data/metadata.tsv')
    print(df)
    print(df.counts_to_df())
    aug_path = make_augments(df, 1)

    train, test, val = Dataset(df, test_size=0.1,
                               val_size=0.05,
                               shuffle=True,
                               random_state=12)
    print(train.shape, test.shape, val.shape)
    cv2.imshow(list(train.take(1))[0]['image'])
