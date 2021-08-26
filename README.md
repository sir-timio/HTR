# Распознавание рукописного текста

Результат работы: https://nbviewer.jupyter.org/github/sir-timio/HTR/blob/main/run.ipynb  

Телеграмм бот: https://t.me/neureader_bot
## Использование с помощью Docker

- установите Docker
- откройте терминал и перейдите в данный каталог
- введите в терминале (это займет несколько минут):
```bash
docker build -t htr/tfgpu .
```
- введите в терминале (измените путь на свой абсолютный путь к репозиторию HTR):
```bash
docker run -it -p 8888:8888 -v /absolute/path/to/HTR:/home/htr htr/tfgpu
```
- скопируйте ссылку в браузер и наслаждайтесь проектом в Jupyter!

# Постановка задачи и актуальность

С каждым днем все больше достижений в области информатики, особенно в машинном обучении и анализе изображений, находят свое применение в реальной жизни. Компьютерное зрение, зарекомендовавшее себя как один из самых эффективных способов анализа изображений и видео, облегчающий или заменяющий работу человека, давно интегрировано во множество сфер деятельности. Распознавание рукописного текста (РРТ) является одной из ключевых задач компьютерного зрения и имеет следующие преимущества:

### Доступ к данным: возможность поиска и удобство использования

РРТ значительно повысил доступность данных. После сканирования и преобразования информации в любой редактируемый формат, такой как MS Word или Adobe PDF, программное обеспечение позволяет хранить или копировать файл в любом месте, где вы хотите, что означает, что файлы затем могут быть найдены в системе вашей компании, и любой человек, имеющий разрешение, может получить к ним доступ. Банковская отрасль и торговые компании получат возможность оптимизировать трудоемкую бумажную работу.

### Экономия времени и памяти

Хранение в облаке - правильный путь, если вы хотите иметь доступную для поиска информацию и сэкономить. Управление бумажной информацией является неэффективной задачей. Этот процесс является одним из самых трудоемких и дорогих. Но самое худшее в этой ручной работе - вероятность человеческой ошибки. Доступ к цифровым данным улучшает рабочие процессы компаний. Используя РРТ, компании могут сократить количество ошибок. Это особенно ценится страховыми компаниями из-за большого количества документов, с которыми им приходится работать каждый день.

### Повышение Удовлетворенности Клиентов

Улучшение качества обслуживания клиентов - это то, к чему стремится каждая компания, и РРТ действительно может помочь в этом. Давайте подумаем о зоне поддержки клиентов, где агенты постоянно получают звонки или электронные письма с запросами. Используя технологию программного обеспечения РРТ, они могут представить себе все услуги, которые клиент имеет в компании, потому что информация доступна в один клик. Это не только уменьшает время, затрачиваемое на каждое дело, но и позволяет службе поддержки клиентов решать любые проблемы, требующие немедленного решения.

### Повышение безопасности

Цифровая среда требует повышения безопасности, особенно для конфиденциальной информации, управляемой полицейскими департаментами, гражданскими учреждениями или обработки персональных данных. Технология РРТ запрограммирована на предотвращение попыток мошенничества путем сравнения предоставленной информации с сохраненными данными с минимумом ошибок, что невозможно сделать вручную.

Технической задачей данного проекта является создание модели для распознавание русского рукописного текста, качество которой оценивается посредством метрик CER и accuracy, а бизнесс задачей — оптимизация документооборота, качество которой можно оценить в денежных единицах на обработку одного изображения с помощью сервисов, доступных на рынке.

# Набор данных

HKR (https://github.com/abdoelsayed2016/HKR_Dataset) — закрытая база данных, содержащая фрагменты русского и казахского рукописного текста. Помимо 33 символов русского алфавита в наборе присутствуют 9 символов казахского алфавита.
HKR представляет собой набор форм. Все формы были созданы с помощью LATEX и впоследствии были заполнены людьми. База данных состоит из более чем 1400 заполненных форм. Всего около 63000 предложений, более 715699 символов, написанных примерно 200 разными авторами.  
Набор данных полностью соотвествует требованиям для решения поставленной задачи, поскольку имеет распределение по символам, аналоличное словарю русского языка.

EDA, предобработка данных и валидация: https://nbviewer.jupyter.org/github/sir-timio/HTR/blob/main/preprocess/Preprocess.ipynb

# Дальнейшие планы

Подключение словаря русского языка (или предметной области) для повышения качества распознавания — при анализе ошибок работы модели на тестовых данных были найдены закономерности, на основе которых можно сделать вывод, что подключение словаря позволит значительно уменьшить CER, соответственно, повысить accuracy.


# Оценка экономического эффекта

При использовании ручного труда:

    n * (profit - r_t * salary_h - err_h * fail_cost)

При использовании технологий сторонних компаний:  

    n * (profit - cost - err_1 * fail_cost)

При использовании собственных технологий:    

    n * (profit - err_2 * fail_cost) - r_ds * r_t * salary_ds

все расчеты производятся на месяц
 - n - количество изображений для обработки. Могут представлять из себя анкеты, документы, заявки и т.п.  
 - profit - прибыль компании от одного обработанного изображения
 - fail_cost - цена ошибки
 - salary_h - з/п в час при ручной обработке
 - salary_ds - з/п в час разработчика технологии
 - r_t - время работы одного человека при ручной обрабатывании
 - r_ds - количество разработчиков (2 < r_ds < 6)
 - r_t_ds - время работы разработчика (r_t_ds < r_t в общем случае)
 - salary_ds - з/п в час разработчика технологии
 - err_h - ошибка при работе человека
 - err_1 - ошибка сторонней технологии (err_h < err_1)
 - err_2 - ошибка нашей технологии (err_h < err_2)

Как можно видеть из формул, затраты первых двух решений растут линейно с увеличением n. При успешном внедрении технологии, err_1 ~ err_2 ~ err_h.
С учетом того, что err_2 принимает малые значения, которые могут становится еще меньше, наша прибыль растет при неизменных затратах. Это позволяет масштабировать бизнес.  
Ошибка err_2 обратно пропорциональна качеству нашей модели, поэтому при увеличении точности на 1 или на 10 процентов, прибыль вырастет на 0.01 и 0.1 стоимости ошибки соответсвенно. Данная гибкость вносит ясность вопрос: стоит улучшать модель или нет.   
# Готовые решения и их стоимость

Сколько стоит распознать текст с одной картинки? В Google и Яндексе — 10 копеек, в Microsoft — 5 копеек при точности в районе 96%. Кажется, не так уж дорого.
Но что, если нужно распознать не одну картинку, а, скажем, десятки миллионов? Например, при объемах 50 млн загружаемых изображений в день, нам бы приходилось тратить от 2.5 млн рублей ежедневно.  
За год получается 1-2 млрд рублей. Это очень дорогое удовольствие для многих компаний.

# Применение РРТ в Российских компаниях

Одним из самых показательных примеров применения распознавания бумажных документов с рукописным текстом – это опыт сети **«Спортмастер»**, где покупателями от руки оформляется сотни тысяч анкет ежемесячно, при этом магазины сети расположены не только по всей России, от Калининграда до Петропавловска-Камчатского, но и странах СНГ и Китае, в сумме это более 400 магазинов.
Внедрение технологии распознавания рукописного текста в анкетах покупателей позволило в 2 раза увеличить скорость обработки этих анкет и полностью отказаться от доставки бумажных экземпляров в центральных офис для обработки.
Важно отметить, что обработка китайских иероглифов оказалась дешевле и качественнее, чем через китайских подрядчиков.

В **«Ситимобил»**: для осуществления задачи по фотоконтролю такси — распознаем госномер автомобиля на фото. Распознанные данные сравниваются с данными в карточке водителя. Таким образом мы определяем правильная машина пришла на контроль или нет.

В сервисе для объявлений **«Юла»**: с помощью OCR мы находим лекарственные препараты, которые запрещены к продаже на Юле. Для этого распознанный текст сравнивается со справочником лекарственных средств. Кроме того, мы определяем номера телефонов, ссылки на сторонние сайты, ник Инстаграма и т. д.

Для рекламной платформы **myTarget**: мы группируем рекламные баннеры по наличию одинакового текста на изображении. Это позволяет сократить количество ручной модерации, а также использовать текстовые классификаторы для определения рекламы низкого качества.



# Заключение

Все поставленные задачи были выполнены: разработана модель для распознавания русского рукописного текста, качество которой на тестовых данных — CER: 4.64%, accuracy: 77.95%, экономический эффект внедрения данной модели был рассмотрен в пункте "Оценка экономического эффекта".
Использование технологий распознавания текста позволяет легко соответствовать внутренним стандартам документооборота и полностью или частично устраняет необходимость в бумажном документообороте. Высокоуровневые услуги оптического распознавания символов могут помочь многим средним и крупным компаниям получить прибыль от использования специально разработанных алгоритмов. Такие отрасли, как банковское дело и финансы, здравоохранение, туризм и логистика, могут извлечь наибольшую выгоду из успешного внедрения РРТ. И с каждым годом потребность такой услуги будет только возрастать.
