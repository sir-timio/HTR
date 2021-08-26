from htr.model.Model import Model
import numpy as np
from PIL import Image
import os

img_width = 900
img_height = 120

# parameters of resized images
new_img_width = 350
new_img_height = 50

batch_size = 16

# default paths
WORKING_DIR = '/home/mts/mysite/htr'
ann_path = os.path.join(WORKING_DIR, 'ann')
img_path = os.path.join(WORKING_DIR, 'img')
metadata = os.path.join(WORKING_DIR, 'metadata', 'metadata.tsv')

model_params = {
    'callbacks': ['checkpoint', 'csv_log', 'tb_log', 'early_stopping'],
    'metrics': ['cer', 'accuracy'],
    'checkpoint_path': os.path.join(WORKING_DIR, 'checkpoints/training_2/cp.ckpt'),
    'csv_log_path': os.path.join(WORKING_DIR, 'logs/csv_logs/log_2.csv'),
    'tb_log_path': os.path.join(WORKING_DIR, 'logs/tb_logs/log2'),
    'tb_update_freq': 200,
    'epochs': 50,
    'batch_size': batch_size,
    'early_stopping_patience': 10,
    'input_img_shape': (new_img_width, new_img_height, 1),
    'vocab_len': 75,
    'max_label_len': 22,
    'chars_path': os.path.join(os.path.split(metadata)[0], 'symbols.txt'),
    'blank': '#',
    'blank_index': 74
}

model = Model(model_params)
model.build()
model.load_weights('../checkpoints/training_2/cp.ckpt')

def model_predict(full_path='as', gpu=True):
    try:
        img = np.array(Image.open(full_path))
        predicted_text = model.predict_img(img)
    except:
        predicted_text = 'something went wrong...'
    return predicted_text
