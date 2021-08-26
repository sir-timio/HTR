import os
import logging
from aiogram import Bot
import asyncio
import time

from bot.config import TOKEN, MY_ID

logging.basicConfig(format=u'%(filename)s [ LINE:%(lineno)+3s ]#%(levelname)+8s [%(asctime)s]  %(message)s',
                    level=logging.DEBUG)


bot = Bot(token=TOKEN)

async def upload(folder='./img', to_write='data/samples.txt'):
    ids = []
    i = 0
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'rb') as f:
            msg = await bot.send_photo(MY_ID, f, disable_notification=True)
            file_id = msg.photo[-1].file_id
            ids.append(file_id)
            i += 1
            if i % 50 == 0:
                time.sleep(11)

    with open(to_write, 'w') as f:
        for i in ids:
            f.write("%s\n" % i)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(upload(folder='./samples'))
