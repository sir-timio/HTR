import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class CTCLayer(layers.Layer):

    def __init__(self, blank_index, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost
        self.blank_index = blank_index

    def get_config(self):
        return super().get_config()

    def cer(self, y_true, y_pred, pred_sequence_length, true_sequence_length):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")

        pred_codes_dense = K.ctc_decode(y_pred, tf.squeeze(pred_sequence_length, axis=-1), greedy=True)
        # -1 - blank in ctc_decode

        pred_codes_dense = tf.squeeze(tf.cast(pred_codes_dense[0], tf.int64), axis=0)  # only [0] if greedy=true
        idx = tf.where(tf.not_equal(pred_codes_dense, -1))
        pred_codes_sparse = tf.SparseTensor(tf.cast(idx, tf.int64),
                                            tf.gather_nd(pred_codes_dense, idx),
                                            tf.cast(tf.shape(pred_codes_dense), tf.int64))

        idx = tf.where(tf.not_equal(y_true, self.blank_index))
        label_sparse = tf.SparseTensor(tf.cast(idx, tf.int64),
                                       tf.gather_nd(y_true, idx),
                                       tf.cast(tf.shape(y_true), tf.int64))
        label_sparse = tf.cast(label_sparse, tf.int64)

        distances = tf.reduce_sum(tf.edit_distance(pred_codes_sparse, label_sparse, normalize=False))

        # compute chars amount represent in y_true
        count_chars = len(idx)

        return tf.divide(tf.cast(distances, tf.float32), tf.cast(count_chars, tf.float32), name='CER')

    def accuracy(self, y_true, y_pred, pred_sequence_length, true_sequence_length):
        batch_len = tf.shape(y_true)[0]

        pred_codes_dense = K.ctc_decode(y_pred, tf.squeeze(pred_sequence_length, axis=-1), greedy=True)
        # -1 - blank in ctc_decode

        pred_codes_dense = tf.squeeze(tf.cast(pred_codes_dense[0], tf.int64), axis=0)  # only [0] if greedy=true
        idx = tf.where(tf.not_equal(pred_codes_dense, -1))
        pred_codes_sparse = tf.SparseTensor(tf.cast(idx, tf.int64),
                                            tf.gather_nd(pred_codes_dense, idx),
                                            tf.cast(tf.shape(pred_codes_dense), tf.int64))

        idx = tf.where(tf.not_equal(y_true, self.blank_index))
        label_sparse = tf.SparseTensor(tf.cast(idx, tf.int64),
                                       tf.gather_nd(y_true, idx),
                                       tf.cast(tf.shape(y_true), tf.int64))
        label_sparse = tf.cast(label_sparse, tf.int64)

        correct_words_amount = len(
            tf.where(tf.equal(tf.edit_distance(pred_codes_sparse, label_sparse, normalize=False), 0))
        )

        return tf.divide(tf.cast(correct_words_amount, tf.float32), tf.cast(batch_len, tf.float32), name='accuracy')

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        cer = self.cer(y_true, y_pred, input_length, label_length)
        accuracy = self.accuracy(y_true, y_pred, input_length, label_length)
        self.add_metric(cer, name='cer')
        self.add_metric(accuracy, name='accuracy')

        return y_pred
