import numpy as np
import telebot
import pickle
import theory

"""x = np.array([-2.52, 4.902, 2.879])
print(network.feedforward(x)) # 0.7216325609518421"""

model = theory.network
bot = telebot.TeleBot('6250290910:AAEqop_yHCZJgeOifJNzpZl8KNUR6I3mlnQ')

perimeter, core_length, core_width = [0, 0, 0]


# периметр	длина ядра	ширина ядра

@bot.message_handler(commands=['start'])
def start_message(message):
    markup = telebot.types.InlineKeyboardMarkup()
    buttonA = telebot.types.InlineKeyboardButton('Да', callback_data='a')
    buttonB = telebot.types.InlineKeyboardButton('Нет', callback_data='b')
    buttonC = telebot.types.InlineKeyboardButton('Не знаю', callback_data='c')

    markup.row(buttonA, buttonB)
    markup.row(buttonC)
    bot.send_message(message.chat.id, 'Привет, хотите определить класс пшеницы?', reply_markup=markup)


@bot.callback_query_handler(func=lambda call: True)
def handle(call):
    if call.data == "a":
        bot.send_message(call.message.chat.id, 'Введите периметр, cm =')
        bot.register_next_step_handler(call.message, get_perimeter)


def get_perimeter(message):
    global perimeter
    perimeter = float(message.text)
    bot.send_message(message.chat.id, 'Введите длину ядра, cm =')
    bot.register_next_step_handler(message, get_core_length)


def get_core_length(message):
    global core_length
    core_length = float(message.text)
    bot.send_message(message.chat.id, 'Введите ширину ядра, cm =')
    bot.register_next_step_handler(message, get_core_width)


def get_core_width(message):
    global core_width
    core_width = float(message.text)
    bot.send_message(message.chat.id, 'Нажмите любую клавишу')
    bot.register_next_step_handler(message, get_predict)


def get_predict(message):
    global perimeter, core_length, core_width
    pred = round(model.feedforward(np.array([perimeter, core_length, core_width])))
    bot.send_message(message.chat.id, f'Данная пшеница {str(pred)} класса')
    print(pred)


bot.polling()