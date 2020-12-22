#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
with open('filename.json') as table:
    data = json.load(table) 
df = pd.DataFrame(data,columns=['url','title','description','store_rating','price','genre','content_rating','downloads','reviews']) 
df


# In[2]:


df=df.drop(176123)


# In[3]:


df['content_rating'].unique()


# In[4]:


df['genre'].unique()


# In[5]:


df=df.drop(columns=['price','downloads'])


# In[6]:


df


# In[7]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import re

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.layers.embeddings import Embedding
from keras.layers.core import SpatialDropout1D
from tensorflow.keras.layers import LSTM
from keras.callbacks import EarlyStopping

from numpy.random import seed
import nltk
from nltk.stem import SnowballStemmer 
import re
from nltk.corpus import stopwords
import spacy
import en_core_web_sm


# In[8]:


df=df.drop(columns=['url'])


# In[9]:


df['text']=df['title']+" "+df['content_rating']+" "+df['genre']+" "+df['description']
df=df.drop(columns=['title','description','content_rating','genre'])


# In[10]:


df['store_rating']=df['store_rating'].apply(lambda x: int(round(x,0)))


# In[11]:


df


# In[12]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_\']')
STOPWORDS = set(stopwords.words('english'))

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

ps = nltk.PorterStemmer()
def stemming(text_sentence): 
    text = [ps.stem(word) for word in text_sentence.split()]
    return ' '.join(text)

wm = nltk.WordNetLemmatizer()
def lemmatize(text_sentence):
    text = [wm.lemmatize(word) for word in text_sentence.split()]
    return ' '.join(text)

def clean_text(text):
    text = re.sub(r'\w*\d\w*', '', text).strip() # removes all words that contains numbers
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = re.sub('\d+', '', text)
    return text

def links_email_remover(text):
    text = re.sub(r"http\S+", "", text) # remove links with protocol
    text = re.sub(r"\w+[.]\S+|\w+[@]", "", text) # remove links without protocol
    return text

def stopwords_remover(text):
    text = re.sub('\'\w+', '', text) # Remove ticks and the next character 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    text = ' '.join(word for word in text.split() if len(word) > 3) # remove stopwors from text 
    return text


"""
links_email_remover
clean_text
lemmatize
stopwords_remover
"""
def preprocess_text(df):
    df = df.reset_index(drop=True)
    df['text'] = df['text'].apply(lambda x: links_email_remover(x))
    print("links removed")
    df['text'] = df['text'].apply(lambda x : clean_text(x))
    print("cleaned")
    df['text'] = df['text'].apply(lambda x : lemmatize(x))
    print("lemmatized")
    df['text'] = df['text'].apply(lambda x: stopwords_remover(x))
    print("stopwords removed")
    df['text'] = df['text'].apply(lambda x : stemming(x))
    print("stemmig")
    df['text'] = df['text'].str.replace('\d+', '')
    print("numbers removed")
    return df


processed_data = preprocess_text(df)
processed_data = processed_data[processed_data['text'].str.len()!=0]
# df_test = preprocess_text(df_test)


# In[13]:


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS, filters='#!"$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split= ' ')
tokenizer.fit_on_texts(processed_data['text'].values)
# Build the word index.
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
word_index['play']


# In[14]:


def fromTextToFeatures(df_text):
    # gives you a list of integer sequences encoding the words in your sentence
    X = tokenizer.texts_to_sequences(df_text.values)
    # split the X 1-dimensional sequence of word indexes into a 2-d listof items
    # Each item is split is a sequence of 50 value left-padded with zeros
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    return X
X = fromTextToFeatures(processed_data['text'])
print('Shape of data tensor:', X.shape)


# In[15]:


Y = pd.get_dummies(processed_data['store_rating']).values
print('Shape of label tensor:', Y.shape)


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[17]:


X


# In[18]:


seed(100)

model = Sequential()
# The Embedding layer is used to create word vectors for incoming words. 
# It sits between the input and the LSTM layer, i.e. 
# the output of the Embedding layer is the input to the LSTM layer.
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary(())

epochs = 3
batch_size = 256 #256

# history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# In[ ]:


history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# In[ ]:


accr = model.evaluate(X_train,Y_train)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

pred_y = model.predict(X_test)


# In[ ]:


accr = model.evaluate(X[2345:2465],Y[2345:2465])
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

pred_y = model.predict(X[2345:2465])


# In[ ]:


pred_y = model.predict(X_test)


# In[ ]:


print(len(df[df['store_rating']==1]))
print(len(df[df['store_rating']==2]))
print(len(df[df['store_rating']==3]))
print(len(df[df['store_rating']==4]))
print(len(df[df['store_rating']==5]))


# In[ ]:


model.save('OurSadModel')


# In[ ]:


org={
    "title":["ORG 2021","Men Jacket Photo Editor","Funny Videos For Tik Tok Musically","One Star App: The Lowest Rated App","FML Official"],
    'description':["Using the pitchbend: • https://youtu.be/T_RAuErJsSM • https://youtu.be/gTM7mpV-fmk Live Style: • https://youtu.be/VxXfx_gV5Qo • https://youtu.be/4xuraY1r_dc Connecting to a MIDI keyboard: • https://youtu.be/CLKaNJO5XOE • https://youtu.be/EA-GmNKn6e8 Features: • Thousands of musical instruments (Multi-sample and Recorded from real instruments) • Thousands of rhythms (Include Intros, Variations, Fills, Break, Endings, and Pads) with Real Chord (Press 3 or more keys) • DNC Instruments with After-touch • A lot of drum-kits (General, Arabic, Persian, ...) • 3 Low-latency methods (in settings) • Connect to a real musical keyboard via USB MIDI cable • Use the phone as a microphone with audio filters • Programmable Sound/Loop Buttons for play Audio files • Strong Windows tools for Create, Edit and Import Instruments, Styles, ... (download from www.sofeh.com) • Joystick, Ribbon, Effects, and Filters • Record, Re-record, Sing a Song, Save, Playback, ... • High-quality Stereo output • Simulate KORG keyboards with: Fade, Synchro, Keyboard Set, Tempo, Transpose, Octave, Balance, Tune, Split, Chord Memory, Tap Tempo/Reset ... • Multi-touch with 10 fingers • Change volume of each Instrument or Style Separately • Pedal, Metronome and Touch Response • Quarter tones (Arabic, Persian, Kurdish ...) • 1 and 2 rows keyboard • Create high-quality MP3 and share on Social Networks ( WhatsApp, Viber, Telegram, Line, ... ) • and more ... This app is also known as ORG 2017, ORG 2018, ORG 2019, and ORG 2020. Website: https://www.sofeh.com Support email: support@sofeh.com https://www.instagram.com/rmn_jafari/ https://www.facebook.com/sofehsunrise https://www.youtube.com/c/sofehsunrise","Do you need a well dressed nice looking photo and you don't want to spend money on an expensive men suit? With this Men Suit Photo Editor 2019 amazing application you can put your face into several nice men suits and man hair style. Men Suit Photo Editor 2020 is a latest men apps and usually men apps free android suit editor application which usually provides HD men suit design dress, glasses, hair styles and mustaches collection. Here you can see editing style how you look in these mens suit blazer coat hd image clothes. Create custom photos with men suit photo montage effects app and edit your photo with 2020 new stylish collection & latest design. Create the Stylish photo montage with men casual shirt photo suit and man hair style using this picture editor app. Man Hair Mustache Style changer makes up kit lets you change your hairstyle and mustache style together. Make handsome and smart hairstyles with editing style photo mustache and beard app. Have fun with Man suit photo editor with Man Hairstyle Editor, hair color change and men glasses editor. Men Suit Photo Camera is your new stylish photo editor men apps that will help you look like fancy and modern. Men casual shirt is a man photo free application to design your photo into impressive frames. Choose photo from gallery or take new photo with camera. Men photo suit is so easy to use, so make photos man hairstyles with nice looking men casual shirt without any hard work. It contains funny, fashionable, Man fashion photo suit,professional suit with professional background,professional Men,Photo Editor, Business Men,men clothes changer app, fashion photo editor,Men Jacket Photo Frame, Jacket Suit, Bisness Suit, professional Suit, Kurta, latest suits in many colors! This Suit Changer Photo Editor is specially made for people who love to edit images and take more photos in fashion men suit. Here you know how to use this app: a) After installation open Men Suit Editor 2018 photo frame app b) Capture photo using picture editor app or select from gallery images c) Select Boys casual and fashion suit, glasses, hairs and mustaches d) Rotate it and fit as you like e) Then save it in device or send friends via social media networks FEATURES: • A lot of beautiful man dress photo suit • Easy and natural application • Change any picture inside a quick one touch • Hairstyle available for your Man Hairstyle • Men sunglasses free available • Install TOTALLY FREE and can be used without Wifi • Save pictures into the individual folder of this app • This particular app can take pictures from the SD card or phone memory space as well • Send these photos with your friends. • Can be use anywhere Enjoy MEN Photo Maker Editor? We look forward for good comments, thank you.","Are you looking for Trending videos? this funny and videos status application you can enjoy the best videos app. With this one app you can watch viral, funny and trending videos and Tiktok Videos. Funny Videos For Musically app because there are many Hindi funny videos from various applications that are most popular today and everyday is perfect for those of you who want to appear hits enjoy funny moments in your life. Free Streaming app for Tiktok & musical.ly Videos. Watch the most hit and new Tiktok funny videos using this best video application. Are you a fan of Hot Video? \"Hot Video For Tik Tok & Social Media\" Is Largest Collection Of Funny | Comedy | Live | Hottest | New Viral | Bhojpuri Hot Videos for Tik Tok Musically & Social Media App You Can Enjoy The Best Videos App. Tik tok one of the best and actual social media apps, which gives an opportunity to users to make funny videos and photos. App Feature: - The app is updated frequently. - Very easy Interface. - Troll video. - Funny Video. - Trending Video. - Love Video. - Trending Video Status. - Love Video Status. - Sad Video Status. - Romantic Video. - Sad Video. - User can also view more application from developers. - Personalize your play list by marking a song as your favorite. - Share video to Facebook, Whatsapp, Twitter, Instagram and other social networks. Disclaimer: We don’t host any of these videos Files, This app is not affiliated with Tik Tok or musical.ly or Dubsmash. All rights reserved to the content's respective owners.","Step 1: Install Step 2: Rate One Star Step 3: Invite Friends Step 4: Uninstall","FML has been around for almost 10 years now (give or take a few years), and we're not going to stop the party yet! Like punk, FML will never die. The idea behind our app is quite simple: every day, our team selects the daily mishaps and embarrassments that our brave users have sent in, to serve them to the rest of you on a plate. Your life isn't perfect either? Let's have a laugh about it! FML is also a social network, the network of people proudly living crappy lives. Don't feel like you have to post fabulously flamboyant stuff here like on the other social networks: you won't be judged by us (well, not that much). No need for perfect photos of fantastic parties that seem a bit too staged and filtered, as well as status updates that seem slightly exaggerated. Thanks to our Timeline, you'll have your own space in which to express yourself (at last) freely, and show everyone how crappy your (real) life is… and show everyone that you're able to laugh it off! Problems with transportation, sports injuries, juicy stories… Perfection is being able to laugh about your own imperfections! Download FML to join our millions of subscribers, meet people (in a friendly way, of course, sort of!), submit FMLs, find out the latest offbeat news with FML News, take part in debates in the comments, check out the funniest photos our team and our users have found, send each other messages via the private messaging system, check out the illustrated FMLs, giggle at our video content… And all this for free! Any problems? Not happy? Tell us all about it at support@viedemerde.fr (instead of here, which would be cooler for us)"],
    'store_rating':[4.3,3.7,2.9,1.6,3.3],
    'genre':["Music & Audio","Personalization","Entertainment","Entertainment","Entertainment"],
    'content_rating':["Rated for 3+","Rated for 3+","Rated for 12+","Rated for 3+","Rated for 12+"]
}
tt=pd.DataFrame.from_dict(org)
tt


# In[ ]:


tt['text']=tt['title']+" "+tt['content_rating']+" "+tt['genre']+" "+tt['description']
tt=tt.drop(columns=['title','description','content_rating','genre'])

tt['store_rating']=tt['store_rating'].apply(lambda x: int(round(x,0)))


# In[ ]:


tt=preprocess_text(tt)


# In[ ]:


smallx=fromTextToFeatures(tt['text'])
smallx


# In[ ]:


smally=np.array([[0,0,0,1,0],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[0,0,1,0,0]])
smally


# In[ ]:


accr = model.evaluate(smallx,smally)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

pred_y = model.predict(smallx)
pred_y

