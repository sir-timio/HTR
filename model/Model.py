import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model.metrics import CTCLayer
from model.Speller import Speller

r'''
params:

callbacks: list of callback names
metrics: list of metric names
checkpoint_path: str path for checkpoints
csv_log_path: str path for csv logs
tb_log_path: str path for files to be parsed by TensorBoard
tb_update_freq: 'batch'/'epoch'/int frequency of writing to TensorBoard
epochs: int number of epochs
batch_size: int size of batch
early_stopping_patience: int early stopping patience
input_img_shape: array(width, height, 1)
vocab_len: int length of vocabulary with blank
max_label_len: int max length of labels
chars_path: path to file that contains alphabet
blank: str blank symbol for ctc
'''

class Model():

    def __init__(self, params):
        self.callbacks = []
        self.epochs = params['epochs']
        self.metrics = params['metrics']
        self.history = dict()

        self.model = None
        self.pred_model = None

        self.input_shape = params['input_img_shape']
        self.batch_size = params['batch_size']
        self.vocab_len = params['vocab_len']
        self.max_label_len = params['max_label_len']
        self.chars_path = params['chars_path']
        self.blank = params['blank']
        self.blank_index = None

        self.vocab = None
        self.num_to_char = None
        self.char_to_num = None
        self.__set_mapping()
        self.speller = Speller(self.vocab, params['corpus'])


        if 'checkpoint' in params['callbacks']:
            self.cp_path = params['checkpoint_path']

            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.cp_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 mode='min',
                                                 verbose=1)
            self.callbacks.append(cp_callback)

        if 'csv_log' in params['callbacks']:
            self.csv_log_path = params['csv_log_path']
            csv_log_callback = tf.keras.callbacks.CSVLogger(self.csv_log_path,
                                                      append=False, separator=';')
            self.callbacks.append(csv_log_callback)

        if 'tb_log' in params['callbacks']:
            self.tb_log_path = params['tb_log_path']
            tb_log_callback = tf.keras.callbacks.TensorBoard(log_dir=self.tb_log_path,
                                                             update_freq=params['tb_update_freq'])
            self.callbacks.append(tb_log_callback)

        if 'early_stopping' in params['callbacks']:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=params['early_stopping_patience'], restore_best_weights=True
            )
            self.callbacks.append(early_stopping)

    def __set_mapping(self):
        self.vocab = open(self.chars_path, encoding="utf8").read().split("\n")
        # Mapping characters to integers
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.vocab, mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        self.blank_index = self.char_to_num(tf.strings.unicode_split(self.blank, input_encoding="UTF-8")).numpy()[0]

    def load_weights(self, path):
        self.model.load_weights(path)

        self.pred_model = keras.models.Model(
            self.model.get_layer(name='image').input, self.model.get_layer(name='dense2').output
        )

    def build(self):
        self.__set_input()
        self.__set_CNN()
        self.__set_RNN()
        self.__set_output()

        self.model = keras.models.Model(
            inputs=[self.input, self.labels], outputs=self.output, name="htr_model"
        )

        # Optimizer
        opt = keras.optimizers.Adam()

        # Compile the model
        self.model.compile(
            optimizer=opt,
        )

    def __set_input(self):
        self.input = layers.Input(
            shape=self.input_shape, name='image', dtype='float32'
        )
        self.labels = layers.Input(name='label', shape=(None, ), dtype='float32')

    def __set_CNN(self):
        self.x = layers.Conv2D(
            32,
            (5, 5),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(self.input)
        self.x = layers.MaxPooling2D((2, 2), name="pool1")(self.x)

        self.x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(self.x)
        self.x = layers.MaxPooling2D((2, 2), name="pool2")(self.x)

        self.x = layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv3",
        )(self.x)
        self.x = layers.MaxPooling2D((2, 2), name="pool3")(self.x)

        self.x = layers.Conv2D(
            256,
            (2, 2),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv4",
        )(self.x)

        new_shape = ((self.input_shape[0] // 8), (self.input_shape[1] // 8) * 256)
        self.x = layers.Reshape(target_shape=new_shape, name="reshape")(self.x)
        self.x = layers.Dense(64, activation="relu", name="dense1")(self.x)
        self.x = layers.Dropout(0.2)(self.x)

    def __set_RNN(self):
        self.x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(self.x)
        self.x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(self.x)

    def __set_output(self):
        self.x = layers.Dense(
            self.vocab_len, activation="softmax", name="dense2"
        )(self.x)
        self.output = CTCLayer(self.blank_index, name="ctc_loss")(self.labels, self.x)

    def get_summary(self):
        return self.model.summary()

    def fit(self, train, val):
        self.history = self.model.fit(
            train,
            validation_data=val,
            epochs=self.epochs,
            callbacks=self.callbacks,
        )

        self.pred_model = keras.models.Model(
            self.model.get_layer(name='image').input, self.model.get_layer(name='dense2').output
        )

        print(f'\n\nmodel weights saved at {self.cp_path}\n\n')

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                  :, :self.max_label_len
                  ]

        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8").replace('[UNK]', "")
            output_text.append(res)

        return output_text

    def predict(self, batch):
        batch_images = batch['image']

        pred = self.pred_model.predict(batch_images)
        pred_texts = self.decode_batch_predictions(pred)

        return pred_texts

    def __strided_rescale(self, img, bin_fac):
        strided = as_strided(img, shape=(img.shape[0]//bin_fac, img.shape[1]//bin_fac, bin_fac, bin_fac),
                             strides=((img.strides[0]*bin_fac, img.strides[1]*bin_fac)+img.strides))
        return strided.mean(axis=-1).mean(axis=-1)

    def __resize_img(self, img, new_img_height, new_img_width):
        img_size = np.array(img.shape[:2])
        new_img_size = np.array([new_img_height, new_img_width])
        diff = img_size - new_img_size
        h_ratio = w_ratio = 0
        if diff[0] > 0:
            h_ratio = img_size[0] / new_img_size[0]
        if diff[1] > 0:
            w_ratio = img_size[0] / new_img_size[0]
        if h_ratio != 0 or w_ratio != 0:
            ratio = round(max(h_ratio, w_ratio))
            img = self.__strided_rescale(img, ratio)
        return img

    def __apply_brightness_contrast(self, input_img, brightness=0, contrast=0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def __encode_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_val = img.mean()
        if mean_val < 230:
            img = self.__apply_brightness_contrast(img, 230/mean_val*60, 230/mean_val*30)
        img = self.__resize_img(img, self.input_shape[1], self.input_shape[0]).astype(np.uint8)
        img = np.expand_dims(img, 2)
        img = tf.convert_to_tensor(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = 1 - img
        img = tf.image.resize_with_pad(img, self.input_shape[1], self.input_shape[0])
        img = 0.5 - img
        img = tf.transpose(img, perm=[1, 0, 2])
        return tf.expand_dims(img, 0)

    def predict_img(self, img): #img is np.ndarray of shape (height, width, 3) / (height, width, 1) - rgb, gray
        try:
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
            img = self.__encode_img(img)
            pred = self.pred_model.predict(img)
            pred_text = self.decode_batch_predictions(pred)
            return self.speller.compute_img(pred_text[0])
        except ValueError:
            return "Error: Incorrect photo"

    def evaluate(self, batch):
        return self.model.evaluate(batch)

    def get_history(self):
        return self.history.history
