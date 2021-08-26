import asyncio
import sys, os
import logging
import random
from aiogram import Bot, types
from aiogram.utils import executor
from aiogram.utils.emoji import emojize
from aiogram.dispatcher import Dispatcher
from aiogram.types.message import ContentType
from aiogram.utils.markdown import text, bold, italic, code
from aiogram.types import ParseMode, InputMediaPhoto, InputMediaVideo, ChatActions
import numpy as np
from PIL import Image
import requests
from bot.config import TOKEN
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton
sys.path.append('..')
from model.Model import Model


img_width = 900
img_height = 120

# parameters of resized images
new_img_width = 350
new_img_height = 50

batch_size = 16

# default paths
print(os.path.abspath(os.getcwd()))
WORKING_DIR = os.path.join('../')
ann_path = os.path.join(WORKING_DIR, 'ann')
img_path = os.path.join(WORKING_DIR, 'img')
metadata = os.path.join(WORKING_DIR, 'metadata', 'metadata.tsv')

logging.basicConfig(format=u'%(filename)s [ LINE:%(lineno)+3s ]#%(levelname)+8s [%(asctime)s]  %(message)s',
                    level=logging.INFO)

samples = np.loadtxt('data/samples.txt', dtype=str, comments="#")
num_samples = len(samples)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

error_process_messages = [
    'что-то в глаз попало :smiling_face_with_tear: ',
    'Я не знаю, что с этим делать :man_shrugging:'
]

prediction_process_messages = [
    'мне кажется, это "@"',
    'похоже на "@"',
    'я вижу в этом "@"',
    'кажется, это "@"',
    # '@[:3]... @!',
    '"@", угадал?'
]

demo_process_messages = [
    'раз сам писать не хочешь, поищу у себя что-нибудь...\nнашел!',
    'понимаю, не у всех бумага с ручкой под рукой\nдержи',
    'передаю запрос на спутник... :thinking_face:',
    'устанавливаю связь с космосом... :thinking_face:',
]

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
    'blank_index': 74,
    'vocab': list('!(),-.:;?АБВГДЕЖЗИЙКЛМНОПРСТУФХЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё #')
}

model = Model(model_params)
model.build()
model.load_weights('../checkpoints/training_2/cp.ckpt')

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    start_kb = ReplyKeyboardMarkup(
        resize_keyboard=True, one_time_keyboard=True
    ).add(KeyboardButton('/help'))
    await message.reply(text(emojize('Привет! Я бот, который любит читать. Умею только на русском, но быстро учусь новому :nerd_face:'
                        '\nИспользуй /help, '
                        'чтобы узнать список доступных команд!')), reply_markup=start_kb, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    ex_b = KeyboardButton('/examples')
    demo_b = KeyboardButton('/demo')
    info_b = KeyboardButton('/info')
    # commands = ['/examples, /demo, /info'] bad idea
    help_kb = ReplyKeyboardMarkup(
        resize_keyboard=True, one_time_keyboard=False
    ).add(ex_b).add(demo_b).add(info_b)

    msg = emojize(text(bold('Присылай фото, а я попробую угадать, что ты там написал\nдоступны следующие команды:'),
               '/examples - пример ожидаемых фото',
               '/demo - проверить нейросеть на случайном изображении',
               '/info - подробная информация о проекте',
               'Возникли трудности? пиши @sir\_timio :smirking_face:', sep='\n'))
    await bot.send_message(message.from_user.id, msg, parse_mode=ParseMode.MARKDOWN, reply_markup=help_kb)


@dp.message_handler(commands=['examples'])
async def process_examples_command(msg: types.Message):
    i = random.randint(0, num_samples-6)
    media = [InputMediaPhoto(samples[i], 'жду что-то похожее')]
    for photo_id in samples[i+1:i+5]:
        media.append(InputMediaPhoto(photo_id))

    await bot.send_media_group(msg.from_user.id, media)

@dp.message_handler(commands=['demo'])
async def process_photo_command(msg: types.Message):
    i = random.randint(0, num_samples)
    if i > int(num_samples * 0.4):
        replica = demo_process_messages[i % len(demo_process_messages)]
        await bot.send_message(msg.from_user.id, text=emojize(replica))
    img_hash = samples[i]
    url = get_path(img_hash)
    img = np.array(Image.open(requests.get(url, stream=True).raw))
    predicted_text = model.predict_img(img)
    caption = prediction_process_messages[i % len(prediction_process_messages)]
    caption = caption.replace('@', predicted_text)
    # use file path to download image via PIL
    #https://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python
    await bot.send_photo(msg.from_user.id, img_hash,
                         caption=emojize(caption))

def get_path(hash):
    LINK = 'https://api.telegram.org/bot<TOKEN>/getFile?file_id=<FILE_ID>'.replace('<TOKEN>', TOKEN)
    PATH = 'https://api.telegram.org/file/bot<TOKEN>/<FILE_PATH>'.replace('<TOKEN>', TOKEN)
    link = LINK.replace('<FILE_ID>', hash)
    r = requests.get(url=link)
    file_path = r.json()['result']['file_path']
    file_path = PATH.replace('<FILE_PATH>', file_path)
    return file_path

@dp.message_handler(commands=['info'])
async def process_info_command(msg: types.Message):
    info = text('код проекта - https://github.com/sir-timio/HTR\n'
                'вебсайт - https://alex123012.github.io/HTR_site/\n')
    await bot.send_message(msg.from_user.id, info)


@dp.message_handler(content_types=['photo'])
async def echo_img(msg: types.Message):
    img_hash = msg.photo[-1]['file_id']
    url = get_path(img_hash)
    # use file path to download image via PIL
    #https://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python
    img = np.array(Image.open(requests.get(url, stream=True).raw))
    predicted_text = model.predict_img(img)
    if str.isupper(predicted_text[0]):
        cases = [str.isupper(s) for s in predicted_text[1:]]
        if any(cases) and not all(cases):
            predicted_text = predicted_text[0] + str.lower(predicted_text[1:])
    i = random.randint(0, len(prediction_process_messages))
    caption = prediction_process_messages[i % len(prediction_process_messages)].replace('@', predicted_text)
    await msg.reply(caption, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(content_types=ContentType.ANY)
async def unknown_message(msg: types.Message):
    i = random.randint(0, len(error_process_messages))
    message_text = text(emojize(error_process_messages[i]),
                        emojize(text(italic('\nЯ просто напомню,'), 'что есть')),
                        code('команда'), '/help')
    await msg.reply(message_text, parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':
    executor.start_polling(dp)