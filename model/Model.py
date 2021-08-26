import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model.metrics import CTCLayer

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
        self.vocab = params['vocab']

        self.num_to_char = None
        self.char_to_num = None
        self.__set_mapping()

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

    def __encode_img(self, img):
        img = tf.convert_to_tensor(img)
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = 1 - img
        img = tf.image.resize_with_pad(img, self.input_shape[1], self.input_shape[0])
        img = 0.5 - img
        img = tf.transpose(img, perm=[1, 0, 2])
        return tf.expand_dims(img, 0)

    def predict_img(self, img):
        img = self.__encode_img(img)

        pred = self.pred_model.predict(img)
        pred_text = self.decode_batch_predictions(pred)

        return pred_text[0]

    def evaluate(self, batch):
        return self.model.evaluate(batch)

    def get_history(self):
        return self.history.history
