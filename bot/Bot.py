import asyncio
import requests
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
from bot.config import TOKEN
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

logging.basicConfig(format=u'%(filename)s [ LINE:%(lineno)+3s ]#%(levelname)+8s [%(asctime)s]  %(message)s',
                    level=logging.INFO)

samples = np.loadtxt('data/samples.txt', dtype=str, comments="#")
num_samples = len(samples)

LINK = 'https://api.telegram.org/bot<TOKEN>/getFile?file_id=<FILE_ID>'.replace('<TOKEN>', TOKEN)
PATH = 'https://api.telegram.org/file/bot<TOKEN>/<FILE_PATH>'.replace('<TOKEN>', TOKEN)
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

error_process_messages = [
    'что-то в глаз попало, узнай, что я умею \help'
]

prediction_process_messages = [
    'мне кажется, это @',
    'похоже на @',
    'я вижу в этом @',
    'кажется, это @',
    # '@[:3]... @!',
    '@, угадал?'
]

demo_process_messages = [
    'ну, раз сам писать не хочешь, поищу у себя что-нибудь...\nнашел!',
    'да уж, не у всех бумажка с ручкой под рукой\nдержи'
]

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    start_kb = ReplyKeyboardMarkup(
        resize_keyboard=True, one_time_keyboard=True
    ).add(KeyboardButton('/help'))

    await message.reply(text(emojize('Привет! Я бот, который любит читать. Умею только на русском, но быстро учусь новому :nerd:'
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

    msg = text(bold('Присылай фото, а я попробую угадать, что ты там написал\nдоступны следующие команды:'),
               '/examples - пример ожидаемых фото',
               '/demo - проверить нейросеть на случайном изображении',
               '/info - подробная информация о проекте',
               'Возникли трудности? пиши @sir\_timio', sep='\n')
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
    label = 'prediction'
    i = random.randint(0, num_samples)
    if i > int(num_samples * 0.7):
        replica = demo_process_messages[i % len(demo_process_messages)]
        await bot.send_message(msg.from_user.id, text=replica)
    caption = prediction_process_messages[i % len(prediction_process_messages)]
    caption = caption.replace('@', label)
    img_hash = samples[i]
    link = LINK.replace('<FILE_ID>', img_hash)
    r = requests.get(url=link)
    file_path = r.json()['result']['file_path']
    file_path = PATH.replace('<FILE_PATH>', file_path)
    # use file path to download image via PIL
    #https://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python
    await bot.send_photo(msg.from_user.id, img_hash,
                         caption=emojize(caption))


@dp.message_handler(commands=['info'])
async def process_info_command(msg: types.Message):
    info = 'https://github.com/sir-timio/HTR'
    await bot.send_message(msg.from_user.id, info)


@dp.message_handler(content_types=['photo'])
async def echo_img(msg: types.Message):
    label = 'предсказание'
    i = random.randint(0, len(prediction_process_messages))
    caption = prediction_process_messages[i].replace('@', label)
    file_id = msg.photo[-1]['file_id']
    await bot.send_photo(msg.from_user.id, file_id, caption=caption)


@dp.message_handler(content_types=ContentType.ANY)
async def unknown_message(msg: types.Message):
    message_text = text(emojize('Я не знаю, что с этим делать :astonished:'),
                        italic('\nЯ просто напомню,'), 'что есть',
                        code('команда'), '/help')
    await msg.reply(message_text, parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':
    executor.start_polling(dp)