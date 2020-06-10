import string 
import numpy as np
import random
import functions
import texts


def run_bot():
    conversation=True
    sp = string.punctuation
    print("BOT: \tЗадай мне правильный вопрос и получишь ответ. Если не знаешь, что я умею, просто спроси меня об этом.")
    print('\tНапример: \"Что ты умеешь?\"')
    while(conversation==True):
        ans=False
        inp = input(prompt='USER: \t')
        res = functions.query_ans(inp)
        inp = "".join([c for c in inp if c not in sp])
        inp=inp.lower().split(" ")
        inp = [w for w in inp if w not in sp]
        if len(set(inp).intersection(set(texts.GREETING_INPUTS))) != 0:
            print("BOT: \t" + random.choice(texts.GREETING_RESPONSES))
            ans = True
        if len(set(inp).intersection(set(texts.THANKS_INPUTS))) != 0:
            print("BOT: \t" + random.choice(texts.THANKS_RESPONSES))
            ans = True
        if len(set(inp).intersection(set(texts.BYE_INPUTS))) != 0:
            print("BOT: \t" + random.choice(texts.BYE_RESPONSES))
            conversation=False
            ans = True
        else:
            if np.max(res) > 0.5:
                if np.argmax(res) == 0:
                    functions.time_output()
                if np.argmax(res) == 1:
                    functions.able_output()
                if np.argmax(res) == 2:
                    functions.programming_output()
                if np.argmax(res) == 3:
                    functions.ml_output()
                if np.argmax(res) == 4:
                    functions.temp_output()
            else:
                if ans == False:
                    print('BOT: \tИзвини, не совсем понял тебя. Попробуй еще раз.')
        print()