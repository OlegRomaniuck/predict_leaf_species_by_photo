import logging
import os

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from DIPLOMA.my_dipl.kursovaya.selective_model import global_predictor

updater = Updater("", use_context=True)
image_from_telegram = 'recived_photo.JPG'
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
dict_leaf = {"Tomato": 1, "Grape": 2, "Apple": 3, "Pepper": 4, 'Strawberry': 5, 'Squash': 6, 'Corn': 7, 'Peach': 8,
             'Soybean': 9, 'Potato': 10, 'Raspberry': 11, 'Cherry': 12, 'Orange': 13, 'Blueberry': 14}
models_list = ["lightgbm", "random_forest", "xgboost1", "xgboost2", "KNN", "CNN"]
CURRENT_MODEL = "CNN"


def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text(
        'Hi! This bot recognize type of leaf by photo\n We support such type of "HEALTH" leafs-spicies: {}\n'
        'Use command: /models  to see list of supported models\n'
        'Use command: /accept_model and pass model name which we be used for leaf'.format(
            dict_leaf))
    print(update)


def models(update, context):
    """Send a message when the command /models is issued."""
    update.message.reply_text('List of supported models: {}. \n Current model is {}'.format(models_list, CURRENT_MODEL))


def accept_model(update, context):
    """Choose what type of model do you want to use """

    user = update.message.from_user
    model = update.message.text
    if "accept_model" in model:
        model = model.split(" ")[1]
    if  model not in models_list:
        update.message.reply_text(
            "Sorry you send or not choose right model, please use command /models to see avaliable models list")
        return
    global CURRENT_MODEL
    CURRENT_MODEL = model
    update.message.reply_text("Model {} from user {} will be accepted as main".format(CURRENT_MODEL, user.first_name ))


def accept_photo(update, context):
    """download foto and calculate typ of species"""
    try:
        # reply_msg = update.message.reply_to_message
        user = update.message.from_user
        logger.info("GET photo from user {} {}".format(user.first_name, user.last_name))
        update.message.photo[-1].get_file().download(image_from_telegram)
        bot = updater.bot
        bot.send_message(update.message.chat.id, "get photo")
        bot.send_message(update.message.chat.id, "please wait until calculation ended")
        predictions = global_predictor( os.path.join(CURRENT_FOLDER, image_from_telegram), CURRENT_MODEL)
        bot.send_message(update.message.chat.id, "REULT IS {}".format(predictions))

    except BaseException as exc:
        print("exception ")
        print(exc.args)


if __name__ == "__main__":
    # Get the dispatcher to register handlers

    # bot.polling()
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("models", models))
    # dp.add_handler(CommandHandler("accept_model", accept_model))
    model_handler = MessageHandler(Filters.text, accept_model)
    dp.add_handler(model_handler)
    photo_handler = MessageHandler(Filters.photo, accept_photo)
    dp.add_handler(photo_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()
