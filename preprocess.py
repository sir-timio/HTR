import pandas as pd
import json, os, glob, string
# from tqdm import tqdm
from sklearn.model_selection import train_test_split
from numpy import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset

def meta_collect(ann_path: str, result_file: str, sep: str='\t') -> None:
    
    '''collect metadata for all images to "result_file"
    from json files in "ann_path" (execution time: about 5 mins)'''

    with open(result_file, 'w',  encoding='utf-8') as f:
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


class PreprocessFrame(pd.DataFrame):
    
    def __init__(self, metadata='metadata.tsv',
                 rem_str='', subs_str='', *args, **kwargs) -> None:
        
        super().__init__(self. __initial_start(metadata), *args, **kwargs)

        self.__rework(rem_str, subs_str)
        
        
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
            except:
                print('Cant open metadata file')
                return 1

    def __rework(self, rem_str: str, subs_str: str) -> None:
        
        ''''''

        if all(self.columns.isin(['predicted', 'isModerated'])):
            self.drop(['predicted', 'isModerated'], axis=1, inplace=True)

        self['description'] = self.description.str.replace('o', 'о').str.replace('H', 'Н')
        self['description'] = self.description.str.replace('–', '-').str.replace('—', '-').str.replace('…', '...')

        for r, s in zip(list(rem_str), list(subs_str)):
            self.description = self.description.str.replace(r, s)

        alphabet_lower = [chr(ord("а") + i) for i in range(32)] + [chr(ord("а") + 33)] # Last is "ё"
        alphabet_upper = [chr(ord("А") + i) for i in range(32)]
        punctuation = list(string.punctuation)

        alphabet = set(alphabet_lower + alphabet_upper + punctuation)
        
        # Creating alphabet from dataset
        counts = self.counts_to_df()
        counts_dict = counts.set_index('symbols')['counts'].to_dict()

        # difference between dataset and reference alphabet
        bad_symbols = set(counts_dict) - alphabet 
        self = self.drop(counts[counts.symbols.isin(bad_symbols)].index.drop_duplicates(), axis=0)

    def counts_to_df(self, column='description') -> pd.DataFrame:
    
        '''Return dataframe with symbols counts in "column"'''

        counts = pd.DataFrame(self[column].str.split('').explode())
        counts = counts.join(counts[column].value_counts(), on=column, rsuffix='1')
        counts.rename(columns={column:'symbols',
                                column + '1':'counts',}, 
                      inplace=True)
        counts = counts[~counts.isin(['', ' '])].dropna()  #.drop_duplicates()
        
        return counts

    def train_test_val_split(self, column:str='description', *args, **kwargs):
        
        ''' '''
        
        counts = self.counts_to_df(column)
        counts.counts = 1
        splitter = counts.reset_index().drop_duplicates().pivot(index='index', columns='symbols').fillna(0)

        train, test, _, ls = train_test_split(self, splitter, *args, **kwargs)

        test, val, _, _ = train_test_split(test, ls, shuffle=True,
                                           test_size=0.33,
                                           random_state=12)

        return PreprocessFrame(train), PreprocessFrame(test), PreprocessFrame(val)
    
    
    
class Dataset(PrefetchDataset):
    
    def __init__(self, df:PreprocessFrame, batch_size:int=16, img_height=None,
                 img_width=None, max_length=None,
                 shuffle_buffer:int=1024,
                 prefetch:int=tf.data.experimental.AUTOTUNE) -> tf.data.Dataset:
        
        self.df = df

        # Constants
        self.img_height = img_height if img_height else self.df.height.max()
        self.img_width = img_width if img_width else self.df.width.max()
        self.max_length = max_length if max_length else self.df.description.apply(len).max()
        
        super().__init__(self.get_dataset(batch_size, shuffle_buffer, prefetch), prefetch)
    
    def get_dataset(self, batch_size:int,
                    shuffle_buffer:int, prefetch:int) -> tf.data.Dataset:

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
        
        ind = self.df.index.tolist()
        random.shuffle(ind)
        self.df = self.df.loc[ind]
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.df.index.to_series().apply(lambda x: os.path.join(img_path, x) + '.jpg').tolist(),
             self.df.description.tolist())
        )

        dataset = (
            dataset.map(
            self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(prefetch)
        )
    
        return dataset

    def encode_single_sample(self, img_path, label):
        
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
        img = 1 - img

        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])

        # 6. Map the characters in label to numbers
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        label = tf.pad(label, [[0, self.max_length-len(label)]], constant_values=self.blank_index)

        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}
    
if __name__ == '__main__':
    ann_path = os.path.join('HKR_Dataset_Words_Public', 'ann')
    img_path = os.path.join('HKR_Dataset_Words_Public', 'img')
    
#     meta_collect(ann_path, 'metadata.tsv')
    
    df = PreprocessFrame(metadata='metadata.tsv')
    print(df)
    print(df.counts_to_df())
    print(df.train_test_val_split(shuffle=True,
                                  test_size=0.15,
                                  random_state=12))
    print(Dataset(df))