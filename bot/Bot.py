import asyncio
import logging

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

EXAMPLES = [
    'AgACAgIAAxkDAANpYSPbzQABKkMms9fy9qm8Fh-snFnaAAJItjEbXn8ZSaTr2gnjem6mAQADAgADeAADIAQ',
    'AgACAgIAAxkDAANoYSPbzdCDjvo12ckN6SGH_rOAJuoAAke2MRtefxlJsUadeGc-6nYBAAMCAAN4AAMgBA',
    'AgACAgIAAxkDAANnYSPbzbJMac4mUQ37r0YswUDR2nEAAka2MRtefxlJTs3Q58lAfwoBAAMCAANtAAMgBA',
]



bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    start_kb = ReplyKeyboardMarkup(
        resize_keyboard=True, one_time_keyboard=True
    ).add(KeyboardButton('/help'))

    await message.reply('Привет! Я бот, который любит читать. Умею только на русском, но быстро учусь новому ::'
                        '\nИспользуй /help, '
                        'чтобы узнать список доступных команд!', reply_markup=start_kb)


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
    media = []
    for photo_id in EXAMPLES:
        media.append(InputMediaPhoto(photo_id))
    await bot.send_media_group(msg.from_user.id, media)

@dp.message_handler(commands=['demo'])
async def process_photo_command(msg: types.Message):
    caption = 'предсказание'

    await bot.send_photo(msg.from_user.id, EXAMPLES[0],
                         caption=emojize(caption))


@dp.message_handler(commands=['info'])
async def process_info_command(msg: types.Message):
    info = 'https://github.com/sir-timio/HTR'
    await bot.send_message(msg.from_user.id, info)


@dp.message_handler(content_types=['photo'])
async def echo_img(msg: types.Message):
    file_id = msg.photo[-1]['file_id']

    await bot.send_photo(msg.from_user.id, file_id, caption='caption')


@dp.message_handler(content_types=ContentType.ANY)
async def unknown_message(msg: types.Message):
    message_text = text(emojize('Я не знаю, что с этим делать :astonished:'),
                        italic('\nЯ просто напомню,'), 'что есть',
                        code('команда'), '/help')
    await msg.reply(message_text, parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':
    executor.start_polling(dp)