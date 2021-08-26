from warnings import filterwarnings
import json
import os
import string
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from numpy import random

import tensorflow as tf
from tensorflow.keras import layers

import imgaug.augmenters as iaa
import imageio

filterwarnings('ignore')


def meta_collect(ann_path: str, result_file: str,
                 sep: str = '\t') -> None:
    """
    All code were made for HKR For Handwritten Kazakh & Russian Database
    (https://github.com/abdoelsayed2016/HKR_Dataset)

    collect metadata for all images from json files

    execution time: about 5 mins

    Parameters
    ----------
    ann_path : str
        Path to directory with annotation json files
    result file : str
        Path to save metadata file
    sep : str, default = '\t'
        separator for metadata file
        (see pandas.DataFrame kwarg 'sep')

    Returns
    -------
    None

    Examples
    --------
    >>> import pandas as pd
    >>> import json
    >>> import preprocess
    >>> with open(sample_json, 'r') as f:
    ...     print(json.load(f))
    ...
    {'size': {'width': 495, 'height': 64}, 'moderation': {'isModerated': 1, 'moderatedBy': 'Norlist', 'predicted': ''}, 'description': 'Шёл человек.', 'name': '0_0_0'}
    >>> preprocess.meta_collect(ann_path=annotation_path,
                                result_file=result_metafile,
                                sep='\t')
    >>> pd.read_csv(result_metafile, sep='\t')
             width  height     description  isModerated moderatedBy  predicted
    0_0_0      495      64    Шёл человек.            1     Norlist        NaN
    0_0_1      494      65     Шёл человек            1     Norlist        NaN
    0_0_10     489      73     Шёл человек            1     Norlist        NaN
    0_0_11     406      46    Шёл человек.            1     Norlist        NaN
    0_0_12     379      76     Шёл человек            1     Norlist        NaN
    ...        ...     ...             ...          ...         ...        ...
    9_9_875    543      94  Вид постоялого            1     Norlist        NaN
    9_9_877    462      73  Вид постоялого            1     Norlist        NaN
    9_9_878    595      83  Вид постоялого            1     Norlist        NaN
    9_9_879    532      72  Вид постоялого            1     Norlist        NaN
    9_9_880    538      72  Вид постоялого            1     Norlist        NaN

    [64943 rows x 6 columns]
    """

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(sep.join(['width', 'height', 'description',
                          'isModerated', 'moderatedBy', 'predicted']) + '\n')

        for file in tqdm(glob(os.path.join(ann_path, '*.json'))):

            with open(file, encoding='utf-8') as js:
                tmp = json.load(js)

            try:
                f.write(sep.join([tmp['name'], str(tmp['size']['width']), str(tmp['size']['height']),
                                  tmp['description'], str(tmp['moderation']['isModerated']),
                                  tmp['moderation']['moderatedBy'], str(tmp['moderation']['predicted'])]) + '\n')
            except Exception:
                print(tmp['description'])


def make_augments(df: pd.DataFrame, WORKING_DIR: str,
                  img_path: str, new_img_width: int, new_img_height: int
                  ) -> pd.DataFrame:
    """
    All code were made for HKR For Handwritten Kazakh & Russian Database
    (https://github.com/abdoelsayed2016/HKR_Dataset)

    make augments for images from "df" to path "aug" in "img_path"

    Parameters
    ----------

    Returns
    -------

    Example
    -------

    """
    paths = df.index.to_series().apply(lambda x: os.path.join(img_path, x) + '.jpg')
    aug_1 = os.path.join(img_path, 'aug_1')

    if not os.path.exists(aug_1):
        os.mkdir(aug_1)

    paths = [paths[i:i + 500] for i in range(0, len(paths), 500)]

    for path in tqdm(paths):
        
        format_seq = iaa.Sequential(
            [
                iaa.CenterPadToFixedSize(width=new_img_width, height=new_img_height, 
                                         pad_mode="constant", pad_cval=(255, 255)),
                iaa.Resize({"height": new_img_height, "width": new_img_width})
            ]
        )

        seq = iaa.Sequential(
            [
                iaa.Sometimes(0.4, iaa.GaussianBlur(3.0)),
                iaa.Sometimes(0.4, iaa.AveragePooling(3)),
                iaa.Sometimes(0.4, iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))),
                iaa.Sometimes(0.4, iaa.GammaContrast((0.5, 1.0))),
                iaa.Invert(0.05, per_channel=True),
                iaa.Sometimes(0.4, iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))),
                iaa.Sometimes(0.4, iaa.SaltAndPepper(0.1)),

                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
                iaa.PerspectiveTransform(scale=(0.02, 0.1))
            ], 
            random_order=True
        )

        img = [imageio.imread(i) for i in path]

        ls = seq(images=format_seq(images=img))
        for i in range(len(path)):
            _, name = os.path.split(path[i])
            name = os.path.join(aug_1, 'aug_' + name)
            cv2.imwrite(name, cv2.cvtColor(ls[i], cv2.COLOR_RGB2BGR))

    aug_df = df.copy()
    aug_df.index = aug_df.index.to_series().apply(lambda x: os.path.join(os.path.split(aug_1)[-1], 'aug_' + x))
    aug_df.to_csv(os.path.join(WORKING_DIR, 'metadata', 'augmeta.tsv'), sep='\t')

    return PreprocessFrame(metadata=aug_df.copy(),
                           img_height=new_img_height, img_width=new_img_width)


class PreprocessFrame(pd.DataFrame):

    def __init__(self, metadata: str = 'data/metadata.tsv', img_height: int = 120,
                 rem_str: str = '', img_width: int = 900,
                 subs_str: str = '', *args, **kwargs) -> None:

        super().__init__(self.__initial_start(metadata), *args, **kwargs)

        self.__rework(rem_str, subs_str, img_height, img_width)

    def __initial_start(self, x):
        """
        All code were made for HKR For Handwritten Kazakh & Russian Database
        (https://github.com/abdoelsayed2016/HKR_Dataset)


        Parameters
        ----------

        Returns
        -------

        Example
        -------

        """

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

    def counts_to_df(self, column: str = 'description') -> pd.DataFrame:
        """
        All code were made for HKR For Handwritten Kazakh & Russian Database
        (https://github.com/abdoelsayed2016/HKR_Dataset)


        Parameters
        ----------

        Returns
        -------

        Example
        -------

        """

        counts = pd.DataFrame(self[column].map(list).explode())
        counts = counts.join(counts[column].value_counts(), on=column, rsuffix='1')
        counts.rename(columns={column: 'symbols',
                               column + '1': 'counts', },
                      inplace=True)
        counts = counts[~counts.isin(['', ' '])].dropna()

        return counts

    def __rework(self, rem_str: str, subs_str: str,
                 img_height: int, img_width: int,) -> None:
        """
        All code were made for HKR For Handwritten Kazakh & Russian Database
        (https://github.com/abdoelsayed2016/HKR_Dataset)


        Parameters
        ----------

        Returns
        -------

        Example
        -------

        """

        if 'predicted' in self.columns:
            self.drop('predicted', axis=1, inplace=True)
        if 'isModerated' in self.columns:
            self.drop('isModerated', axis=1, inplace=True)

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
        self.drop(counts[counts.symbols.isin(bad_symbols)].index.drop_duplicates(), axis=0, inplace=True)
        self['width'] = self['width'].astype(int)
        self['height'] = self['height'].astype(int)
        self.drop(self[(self['width'] > img_width) | (self['height'] > img_height)].index, axis=0, inplace=True)

    def train_test_val_split(self, test_size: float, val_size: float,
                             column: str = 'description', *args, **kwargs):
        """
        All code were made for HKR For Handwritten Kazakh & Russian Database
        (https://github.com/abdoelsayed2016/HKR_Dataset)


        Parameters
        ----------

        Returns
        -------

        Example
        -------

        """

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


class Dataset:

    def __init__(self, df: PreprocessFrame, test_size: float, val_size: float, img_path: str,
                 WORKING_DIR: str, batch_size: int = 16, img_height=120, img_width=900,
                 new_img_height=50, new_img_width=350, aug_df=None, max_length=None,
                 shuffle_buffer: int = 1024, train_test_split=True,
                 prefetch: int = tf.data.experimental.AUTOTUNE, *args, **kwargs) -> None:

        self.df = df
        # Constants
        self.img_height = img_height
        self.img_width = img_width
        self.max_length = max_length if max_length else self.df.description.str.len().max()
        self.aug_df = aug_df if isinstance(aug_df, (pd.DataFrame, str)) else None
        self.new_img_height = new_img_height
        self.new_img_width = new_img_width
        self.iterator_ = self.__get_dataset(batch_size=batch_size, shuffle_buffer=shuffle_buffer, prefetch=prefetch,
                                            test_size=test_size, val_size=val_size, aug_df=aug_df,
                                            train_test_split=train_test_split,
                                            img_path=img_path, WORKING_DIR=WORKING_DIR, *args, **kwargs)

    def __iter__(self):
        return self.iterator_

    def __repr__(self):
        return self

    def __get_dataset(self, batch_size: int, test_size: float, train_test_split: bool,
                      shuffle_buffer: int, prefetch: int, aug_df: pd.DataFrame,
                      val_size: float, img_path: str, WORKING_DIR: str, *args, **kwargs) -> tf.data.Dataset:
        """
        All code were made for HKR For Handwritten Kazakh & Russian Database
        (https://github.com/abdoelsayed2016/HKR_Dataset)


        Parameters
        ----------

        Returns
        -------

        Example
        -------

        """

        # Creating mappers

        # Mapping characters to integers
        counts = self.df.counts_to_df()
        counts = counts.symbols.sort_values().unique().tolist() + [' ', '#']
        vocab = pd.Series(counts).str.encode('utf8')

        with open(os.path.join(WORKING_DIR, 'metadata', 'symbols.txt'), 'w') as f:
            for sym in pd.Series(counts).iloc[:-1]:
                f.write(sym + '\n')
            f.write(pd.Series(counts).iloc[-1])

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

        if train_test_split:
            train, test, val = self.df.train_test_val_split(test_size=test_size,
                                                            val_size=val_size,
                                                            *args, **kwargs)
            list_df = [train, test, val]
        else:
            list_df = [self.df]

        if isinstance(self.aug_df, pd.DataFrame):
            list_df[0] = pd.concat([list_df[0], self.aug_df])

        elif isinstance(self.aug_df, str) and self.aug_df:
            self.aug_df = list_df[0].copy()
            self.aug_df.index = self.aug_df.index.to_series().apply(lambda x: os.path.join('aug_1', 'aug_' + x))
            list_df[0] = pd.concat([list_df[0], self.aug_df])

        for tmp in list_df:
            tmp = tmp.copy()
            ind = tmp.index.tolist()
            random.shuffle(ind)
            tmp = tmp.loc[ind]
            tmp = tf.data.Dataset.from_tensor_slices(
                (tmp.index.to_series().apply(lambda x: os.path.join(img_path, x) + '.jpg').tolist(),
                 tmp.description.tolist())
            )

            tmp = (
                tmp.map(
                    self.__encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
                .batch(batch_size)
                .prefetch(prefetch)
            )

            yield tmp

    def __encode_single_sample(self, img_path: str, label: str) -> dict:
        """
        All code were made for HKR For Handwritten Kazakh & Russian Database
        (https://github.com/abdoelsayed2016/HKR_Dataset)


        Parameters
        ----------

        Returns
        -------

        Example
        -------

        """

        # 1. Read
        img = tf.io.read_file(img_path)

        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)

        # 3. Convert to float32 in [0, 1] range and binarize
        img = tf.image.convert_image_dtype(img, tf.float32)

        # 4. Resize to the desired size
        img = 1 - img
        img = tf.image.resize_with_pad(img, self.new_img_height, self.new_img_width)
        img = 0.5 - img

        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])

        # 6. Map the characters in label to numbers
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        label = tf.pad(label, [[0, self.max_length - len(label)]], constant_values=self.blank_index)

        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}


def main():

    # image sizes
    img_width = 900
    img_height = 120

    # parameters of resized images
    new_img_width = 350
    new_img_height = 50

    # default paths
    WORKING_DIR = os.path.join('/home', 'htr')
    ann_path = os.path.join(WORKING_DIR, 'HKR_Dataset_Words_Public', 'ann')
    img_path = os.path.join(WORKING_DIR, 'HKR_Dataset_Words_Public', 'img')

    # collect metadata
    meta_collect(ann_path, os.path.join(WORKING_DIR, 'metadata', 'metadata.tsv'))

    # get preprocessed metadata dataframe
    df = PreprocessFrame(metadata=os.path.join(WORKING_DIR, 'metadata', 'metadata.tsv'),
                         img_height=img_height, img_width=img_width)
    print(df.shape)

    # Make augments file (if they exists: comment or delete line)
    aug_df = None
    aug_df = make_augments(df=df, img_path=img_path, WORKING_DIR=WORKING_DIR,
                           img_height=img_height, img_width=img_width)

    # get augments metadata dataframe from original dataframe if not starting make_augments
    if not isinstance(aug_df, pd.DataFrame):
        aug_df = df.copy()
        aug_df.index = aug_df.index.to_series().apply(lambda x: os.path.join('aug_1', 'aug_' + x))

    # split data and get datasets
    dataset = Dataset(df, aug_df=aug_df,
                      test_size=0.1,
                      val_size=0.05,
                      img_path=img_path,
                      img_height=img_height,
                      img_width=img_width,
                      new_img_height=new_img_height,
                      new_img_width=new_img_width,
                      WORKING_DIR=WORKING_DIR,
                      shuffle=True,
                      random_state=12)
    train, test, val = list(dataset)
    print(train)


if __name__ == '__main__':
    main()
