import datetime
import random
import requests
import string # to process standard python strings
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pymorphy2
import texts
import numpy as np

def time_output():
    print('BOT:\t' + random.choice(texts.TIME_RESPONSES), datetime.datetime.now().strftime("%H:%M:%S"))
    
def able_output():
    print('BOT:\tЯ - узкоспециализированный чат-бот.')
    print('\tВот что я умею делать:\n\t1. Называю время\n\t2. Рассказываю про свои возможности')
    print('\t3. Советую курсы или литературу по программированию (для краткости можно писать \'прог\')')
    print('\t4. Советую курсы или литературу по data science (для краткости можно писать \'мл\')')
    print('\t5. Говорю, какая сейчас погода в Новосибирске')
    print('\tТакже я умею здороваться и прощаться. Люблю, когда меня благодарят :)')
    
def programming_output():
    print('BOT:\t'+random.choice(texts.PROGRAMMING_RESPONSES_START))
    print('\t'+random.choice(texts.PROGRAMMING_RESPONSES_BOOKS))

def ml_output():
    print('BOT:\t'+random.choice(texts.PROGRAMMING_RESPONSES_START))
    print('\t'+random.choice(texts.ML_RESPONSES_BOOKS))

def temp_output():
    city_id = 1496747
    appid = "05844bc11f55d22c385356767203691e"

    try:
        res = requests.get("http://api.openweathermap.org/data/2.5/weather",
                     params={'id': city_id, 'units': 'metric', 'lang': 'ru', 'APPID': appid})
        data = res.json()
        weather_description = data['weather'][0]['description']
        weather_description = weather_description[0].upper() + weather_description[1:]
        temp = data['main']['temp']
        print('Погода на', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'в Новосибирске')
        print("{}, {}℃.".format(weather_description,temp))
    except Exception as e:
        print("Exception (weather):", e)
        pass
    
def nlp_preproc(inputs):
    ret = inputs.copy()
    tokenizer = RegexpTokenizer(r'\s+',gaps=True)
    sp = string.punctuation
    sw = set(stopwords.words('russian'))
    morph = pymorphy2.MorphAnalyzer()
    for i in range(len(inputs)):
        ret[i] = ret[i].lower()
        ret[i] = "".join([c for c in ret[i] if c not in sp and not c.isdigit()])
        ret[i] = tokenizer.tokenize(ret[i])
        ret[i] = [w for w in ret[i] if w not in sw]
        ret[i] = [morph.parse(w)[0][2] for w in ret[i]]
        ret[i] = [w for w in ret[i] if w not in texts.STOP_WORDS]
        ret[i] = " ".join(ret[i])
    return list(set(ret))

def tokenize_and_to_matrix(inp):
    inp = inp[0].split(" ")
    res = np.zeros(len(texts.word_index)+1)
    for w in inp:
        ind = texts.word_index.get(w,0)
        res[ind] += 1
        res[ind] /= res[ind]
    res = res[1:].reshape(1, len(texts.word_index))
    return res
import model

def query_ans(inp):
    inp = nlp_preproc([inp])
    #print(inp)
    inp = tokenize_and_to_matrix(inp)
    return model.xgb.predict_proba(inp)



