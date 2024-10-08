# CV-T-Bank-Sirius-ML-2024
## Решение отборочного этапа на программу от Т-Банка и Университета Сириус

### Сегментация:
Модель `SAM (Segment Anything Model)` - выбрала именно ее, т.к. она считается SOTA в задаче Zero-Shot Instance Segmentation. Собственно, использую ее тоже в Zero-Shot. Если бы данные были в виде "картинка на каком-то однородном фоне", можно было бы воспользоваться классическими алгоритмами иил попробовать дообучать нейросетки, но конкретно в этом случае у некоторых изображений фон отличается, поэтому решила использовать то, что работает хорошо и дополнительно обучать не нужно.

Используется режим `segment anything` - модель предсказывает маски всех предметов/объектов на фотографии. После этого выбираю нужную маску (автоматически).

### Генерация фона
На самом деле можно было бы скачать пару разных картинок фона и накладывать выделенный объект на выбранный фон. 

Я использовала технологию Inpainting, модель - `dreamshaper-8`. Пробовала разные (Stable Diffusion, Kandinsky 2.2, SDXL, Stable Diffusion 2), но именно выбранная генерировала качественные фоны, остальные добавляли лишние предметы. Реализации выбора цвета фона нет, но она легко добавляется путем изменения промпта.

Преимущество этого метода в том, что не нужно беспокоиться насчет того, что изображение, соединенное с изображением фона, выглядит неестественно.

### Генерация описания
Я использовала модель `BLIP-2`, а затем переводила сгенерированное описание с помощью `Helsinki-NLP/opus-mt-en-ru`. Была идея дообучить BLIP-2 на русском датасете (заморозить веса image encoder'а и поменять изначальную llm на русскую модель, ее веса тоже заморозить, и дообучать только Q-Former), но не хватило времени, поэтому довольствуемся готовыми решениями. Пробовала какие-то image2text модели от сообщества, но они либо плохо генерировали текст, либо не влезали в память.

Если продолжать разрабатывать и улучшать пайплайн, я бы поработала над созданием фона, т.к. хотелось бы протестировать другие методы, а также улучшила бы генерацию текстовых описаний.

## Пример
![mask](https://github.com/user-attachments/assets/cfcc1870-98c8-457b-afa3-68b6cfe19597)
![gen](https://github.com/user-attachments/assets/9445ab4c-3243-4a56-8c39-64da2c19c4ab)

Картинка 1: выделение фона = 1 - выделение объекта

Картинка 2: картинка со сгенерированным фоном

Текст: `футбольный мяч со словом Торрес на нем`

`result.zip` - еще больше примеров
