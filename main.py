import logging
from io import BytesIO
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters, CallbackContext, ConversationHandler
import requests


# Включаем логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = 'http://localhost:5000'  # URL вашего Flask Models

# Функция для обработки команды /start
def start(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info(f"{user.first_name}: Start")

    context.user_data['chat_id'] = update.message.chat_id
    context.user_data['api'] = None

    keyboard = [
        [InlineKeyboardButton("Brain Tumor", callback_data='brain_tumor')],
        [InlineKeyboardButton("Pneumonia", callback_data='pneumonia')],
        [InlineKeyboardButton("Heart Attack", callback_data='heart_attack')],
        [InlineKeyboardButton("Diabetes", callback_data='diabetes')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Выберите тип модели:', reply_markup=reply_markup)


# Функция для обработки выбора Models
def api_choice(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()
    context.user_data['api'] = query.data  # Сохраняем выбранный Models
    print(query.data)
    if query.data == 'brain_tumor':
        query.edit_message_text(text=f"Вы выбрали Brain Tumor. Пожалуйста, отправьте изображение МРТ головного мозга.")
    elif query.data == 'pneumonia':
        query.edit_message_text(text="Вы выбрали Pneumonia. Пожалуйста, отправьте рентгеновский снимок.")
    elif query.data == 'heart_attack':
        query.edit_message_text(text="Вы выбрали Heart Attack. Пожалуйста, отправьте список параметров через запятую, без пробелов.")
        text = 'Возраст (Age) - в годах' + '\n' + \
        'Пол (Sex) - M: Мужской, F: ' + '\n' + \
        'Тип боли в груди (ChestPainType) - TA: Типичная стенокардия, ATA: Атипичная стенокардия, NAP: Не стенокардическая боль, ASY: Ассимптоматическая' + '\n' + \
        'Артериальное давление в покое (RestingBP) - в мм рт. ст.' + '\n' + \
        'Холестерин (Cholesterol) - уровень в сыворотке [мм/дл]' + '\n' + \
        'Уровень сахара в крови натощак (FastingBS) - 1: если FastingBS > 120 мг/дл, 0: в противном случае' + '\n' + \
        'Результаты ЭКГ в покое (RestingECG) - Normal: Нормально, ST: наличие аномалий волны ST-T (инверсии T-волны и/или подъем или падение ST > 0.05 мВ), LVH: вероятная или определенная гипертрофия левого желудочка по критериям Эстеса' + '\n' + \
        'Максимальная частота сердечных сокращений (MaxHR) - числовое значение от 60 до 202' + '\n' + \
        'Стенокардия при физической нагрузке (ExerciseAngina) - Y: Да, N: Нет' + '\n' + \
        'Старый пик (Oldpeak) - числовое значение, измеренное в депрессии' + '\n' + \
        'Наклон ST-сегмента (ST_Slope) - Up: восходящий, Flat: плоский, Down: нисходящий'
        query.message.reply_text(text)
    elif query.data == 'diabetes':
        query.edit_message_text(text="Вы выбрали Diabetes. Пожалуйста, отправьте список параметров через запятую.")
        text = 'Беременности (Pregnancies): количество беременностей' + '\n' + \
        'Глюкоза (Glucose): уровень глюкозы в крови' + '\n' + \
        'Артериальное давление (BloodPressure)' + '\n' + \
        'Толщина кожи (SkinThickness)' + '\n' + \
        'Инсулин (Insulin): уровень инсулина в крови' + '\n' + \
        'Индекс массы тела (BMI)' + '\n' + \
        'Функция диабетического предрасположения (DiabetesPedigreeFunction)' + '\n' + \
        'Возраст (Age)'
        query.message.reply_text(text)
    else:
        query.edit_message_text(text="Ошибочка, попробуйте ещё раз :(")
        start(update, context)


def handle_image(update: Update, context: CallbackContext) -> None:
    api = context.user_data.get('api')
    try:
        file = update.message.photo[-1].get_file()  # Получаем наибольшее качество изображения
        image_stream = BytesIO()
        file.download(out=image_stream)  # Download the image into the BytesIO object
        image_stream.seek(0)

        response = requests.post(f'{API_URL}/predict_image/{api}', files={'file': image_stream})
        if response.status_code == 200:
            prediction = response.json().get('predicted_class')
            probabilities = response.json().get('probabilities')
            update.message.reply_text(f'Предсказанный класс: {prediction}')
            update.message.reply_text('Вероятности принадлежности к классам: \n'+ "\n".join(f"{key}: {value}" for key, value in probabilities.items()))
        else:
            update.message.reply_text('Ошибка при получении предсказания.')
            context.user_data['api'] = None
            start(update, context)
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте еще раз.")
        start(update, context)
    context.user_data['api'] = None

def handle_text(update: Update, context: CallbackContext) -> None:
    api = context.user_data.get('api')
    print(api)
    if api == 'heart_attack':
        try:
            text = update.message.text.split(',')
            params = [float(item) if item.replace('.', '', 1).isdigit() else item for item in text]
            error = validate_parameters_heart(params)

            if error is None:
                response = requests.post(f'{API_URL}/predict_params/heart_attack', json={'parameters': params})
                print(response.text)

                if response.status_code == 200:
                    prediction = response.json().get('predicted_class')
                    probabilities = response.json().get('probabilities')
                    update.message.reply_text(f'Предсказанный класс: {prediction}')
                    update.message.reply_text('Вероятности принадлежности к классам: \n'+ "\n".join(f"{key}: {value}" for key, value in probabilities.items()))
                else:
                    update.message.reply_text('Ошибка при получении предсказания.')
                    context.user_data['api'] = None
            else:
                update.message.reply_text(error+' Пожалуйста, попробуйте еще раз.')
        except Exception as e:
            logger.error(f"Ошибка при обработке параметров: {e}")
            update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте еще раз.")
    elif api == 'diabetes':
        try:
            text = update.message.text.split(',')
            try:
                params = [float(item) for item in text]

                response = requests.post(f'{API_URL}/predict_params/diabetes', json={'parameters': params})
                print(response.text)

                if response.status_code == 200:
                    prediction = response.json().get('predicted_class')
                    probabilities = response.json().get('probabilities')
                    update.message.reply_text(f'Предсказанный класс: {prediction}')
                    update.message.reply_text('Вероятности принадлежности к классам: \n'+ "\n".join(f"{key}: {value}" for key, value in probabilities.items()))

                else:
                    update.message.reply_text('Ошибка при получении предсказания.')
                    context.user_data['api'] = None
            except:
                update.message.reply_text('Все значения должны быть числовые. Пожалуйста, попробуйте еще раз.')

        except Exception as e:
            logger.error(f"Ошибка при обработке параметров: {e}")
            update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте еще раз.")
    else:
        update.message.reply_text("Пожалуйста, сначала выберите тип Models.")
    context.user_data['api'] = None
    start(update, context)

def validate_parameters_heart(params):
    if len(params) != 11:
        return "Ошибка: Входной список должен содержать ровно 11 элементов."
    age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope = params

    if not isinstance(age, (int, float)) or age < 0:
        return "Ошибка: Age должен быть неотрицательным числом."

    if sex not in ['M', 'F']:
        return "Ошибка: Sex должен быть 'M' (Мужской) или 'F' (Женский)."

    if chest_pain_type not in ['TA', 'ATA', 'NAP', 'ASY']:
        return "Ошибка: ChestPainType должен быть одним из: TA, ATA, NAP, ASY."

    if not isinstance(resting_bp, (int, float)) or resting_bp <= 0:
        return "Ошибка: RestingBP должен быть положительным числом."

    if not isinstance(cholesterol, (int, float)) or cholesterol < 0:
        return "Ошибка: Cholesterol должен быть неотрицательным числом."

    if fasting_bs not in [0, 1]:
        return "Ошибка: FastingBS должен быть 0 или 1."

    if resting_ecg not in ['Normal', 'ST', 'LVH']:
        return "Ошибка: RestingECG должен быть одним из: Normal, ST, LVH."

    if not isinstance(max_hr, (int, float)) or not (60 <= max_hr <= 202):
        return "Ошибка: MaxHR должен быть числом в диапазоне от 60 до 202."

    if exercise_angina not in ['Y', 'N']:
        return "Ошибка: ExerciseAngina должен быть 'Y' (Да) или 'N' (Нет)."

    if not isinstance(oldpeak, (int, float)):
        return "Ошибка: Oldpeak должен быть числом."

    if st_slope not in ['Up', 'Flat', 'Down']:
        return "Ошибка: ST_Slope должен быть одним из: Up, Flat, Down."
    return None

# Функция для обработки ошибок
def error(update: Update, context: CallbackContext) -> None:
    logger.warning(f'Обработка ошибки {context.error}')

def help(update: Update, context: CallbackContext) -> None:
    logger.warning(f'Обработка ошибки {context.error}')

def info(update: Update, context: CallbackContext) -> None:
    text = 'Чтобы увидеть информацию по используемым моделям, перейдите по ссылкам' + '\n' + \
        'всё о свертках для обработки изображений - ' + '\n' + \
        'всё о моделях классификаций - '
    update.message.reply_text(text)

def main() -> None:
    updater = Updater("7358756377:AAFtI9_9Qgp4tehHzv06d1ioQuAoauJO0qc", use_context=True)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help))
    dispatcher.add_handler(CommandHandler("info", info))
    dispatcher.add_handler(CallbackQueryHandler(api_choice))
    dispatcher.add_handler(MessageHandler(Filters.photo, handle_image))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))
    dispatcher.add_error_handler(error)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()







