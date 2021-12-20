"""helping source: https://colab.research.google.com/drive/1m8CATHKHro7vJitxKXm_X00Kc899Jdata5
"""

import pandas as pd
import seaborn as sns
import re
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
# nltk.download('wordnet')
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from imblearn import under_sampling
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from PIL import Image, ImageTk
import pickle


# preprocess method
def preprocess(data):
    # preprocess
    a = re.sub("[^a-zA-Z]", ' ', data)
    a = a.lower()
    a = a.split()
    a = [wo.lemmatize(word) for word in a]
    a = ' '.join(a)
    return a

# action taken when send button is pressed
def send():
    text = ent.get("1.0", 'end-1c')
    tk.Label(frm1, text=text, justify=LEFT, wraplength=400,
             font=('Comic Sans MS', 10), bg='#0093E9', fg='white').pack(pady=(0, 7), padx=10, anchor="e")
    ent.delete("1.0", END)
    # button.config(state=tk.DISABLED)
    a = preprocess(text)
    example_counts = vectorizer.transform([a])
    prediction = mnb.predict(example_counts)
    # in case of positive/neutral
    if prediction[0] == 0:
        tk.Label(frm1, text='You seem to be really content', justify=LEFT, wraplength=300,
                 font=('Comic Sans MS', 10), bg='white', fg='black', ).pack(pady=(0, 7), padx=10, side=TOP, anchor="w")
        tk.Label(frm1, text='It was nice talking to you. You can chat '
                            'with me anytime you want.\nBye:)', justify=LEFT, wraplength=350,
                 font=('Comic Sans MS', 10), bg='white', fg='black', ).pack(pady=(0, 7), padx=10, side=TOP, anchor="w")
    # in case of depressed
    elif prediction[0] == 1:
        tk.Label(frm1, text='Don\'t be hard on yourself. I can list some really cool '
                            'ways to handle it.', justify=LEFT, wraplength=350,
                 font=('Comic Sans MS', 10), bg='white', fg='black', ).pack(pady=(0, 7), padx=10, side=TOP, anchor="w")
        tk.Label(frm1, text='You should develop healthy responses which '
                            'include doing regular exercise and taking good quality sleep.', justify=LEFT,
                 wraplength=350,
                 font=('Comic Sans MS', 10), bg='white', fg='black', ).pack(pady=(0, 7), padx=10, side=TOP, anchor="w")
        tk.Label(frm1, text='Be as different as you truly are, get to know yourself at a deep level, '
                            'esteem your individuality, interact with pepole honestly, and eventually the people'
                            ' who appreciate '
                            'you will notice and be drawn in.', justify=LEFT,
                 wraplength=350,
                 font=('Comic Sans MS', 10), bg='white', fg='black', ).pack(pady=(0, 7), padx=10, side=TOP, anchor="w")
        tk.Label(frm1, text='You should have clear boundaries between your work or academic '
                            'life and home life so you make sure that you don\'t mix them.', justify=LEFT,
                 wraplength=350,
                 font=('Comic Sans MS', 10), bg='white', fg='black', ).pack(pady=(0, 7), padx=10, side=TOP, anchor="w")
        tk.Label(frm1, text='Techniques such as meditation and deep breathing exercises can be '
                            'really helping in relieving stress. Always take time to '
                            'recharge so as to avoid the negative thoughts '
                            'and burnout.', justify=LEFT,
                 wraplength=350,
                 font=('Comic Sans MS', 10), bg='white', fg='black', ).pack(pady=(0, 7), padx=10, side=TOP, anchor="w")
        tk.Label(frm1, text='These situations arise in everyone\'s life and what matters the'
                            ' most is taking the right decision at such moments.', justify=LEFT,
                 wraplength=350,
                 font=('Comic Sans MS', 10), bg='white', fg='black', ).pack(pady=(0, 7), padx=10, side=TOP, anchor="w")


# creating the GUI
window = tk.Tk()
window.geometry('400x330')  # dimensions (size) of the window
window.title('Depression Analysis')  # title of the window
window.iconphoto(False, PhotoImage(file='images\\caregiver.png'))  # logo of the window
# window.resizable(0,0)
# create a scrollbar using Canvas class
canv = tk.Canvas(window, bg='#f1f1f1',
                 highlightthickness=0)  # canvas instance with white background and 0 highlight ring
scrollbar = Scrollbar(window, orient="vertical", command=canv.yview)  # create a scrollbar & set its direction
scrollable_frame = tk.Frame(canv,
                            )  # main frame which contains all frames& widgets (except the buttons frame)
scrollable_frame.bind("<Configure>", lambda e: canv.configure(scrollregion=canv.bbox("all")))  # bind the frame
canv.create_window((0, 0), window=scrollable_frame, anchor="nw")  # set the position
canv.configure(yscrollcommand=scrollbar.set)  # when the canvas y-position changes, the scrollbar moves

# frame 1 (holds the chat)
frm1 = tk.Frame(scrollable_frame, relief=FLAT, borderwidth=3)
frm1.pack(fill=tk.BOTH)  # fill the frame in the top of the window
#  labels placed in frame 1
tk.Label(frm1, width=53).pack(side=TOP)
tk.Label(frm1, text='Hello! Thanks for coming here\nPeople say that '
                    'I am a kind and approachable bot.\nHope I can afford help.', justify=LEFT, wraplength=400,
         font=('Comic Sans MS', 10), bg='white', fg='black').pack(pady=(0, 7), padx=10, side=TOP, anchor="w")
tk.Label(frm1, text='Can you tell me how you feel?', justify=LEFT,
         font=('Comic Sans MS', 10), bg='white', fg='black', ).pack(pady=(0, 7), padx=10, side=TOP, anchor="w")

frm3 = tk.Frame(window, relief=FLAT, borderwidth=1, padx=2)  # not included in the scrollable_frame
frm3.pack(fill=tk.BOTH, side=tk.BOTTOM)

# Creating the text area and send widget
ent = Text(frm3, width=43, height=1.5, wrap=WORD)
ent.pack(side=tk.LEFT)
image_send = Image.open('images\\send.png')  # set an icon to the button
image_send = image_send.resize((20, 20), Image.ANTIALIAS)  # resize the image to fit in the button
photo_send = ImageTk.PhotoImage(image_send)
button = Button(frm3, text="", image=photo_send, width=10, command=send)
button.pack(side=tk.LEFT, pady=3, padx=3)

#######################################################################################################################

# reading dataset from the file
data = pd.read_csv("depression_data.csv")

# dropping unnecessary column
data = data.drop('Unnamed: 0', axis=1)

# cleaning the dataset from extra unnecessary characters
wo = WordNetLemmatizer()
corpus = []
for i in range(0, len(data)):
    message = re.sub('[^a-zA-Z]', ' ', data['message'][i])
    message = message.lower()
    message = message.split()
    message = [wo.lemmatize(word) for word in message]
    message = ' '.join(message)
    corpus.append(message)

# split the dataset into training and testing set with 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(corpus, data['label'], test_size=0.3, random_state=2)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=15000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
x_resample, y_resample = SMOTE().fit_resample(X_train_vect, y_train)
x_test_resample, y_test_resample = SMOTE().fit_resample(X_test_vect, y_test)

# applying naive bayes multinational classifier
mnb = MultinomialNB()
mnb.fit(x_resample, y_resample)

# reporting classifier performance
y_pred1 = mnb.predict(x_resample)
print('*****************************************************')
print('\t\t\tReport Training Set Results')
print('*****************************************************')
print(classification_report(y_resample, y_pred1))
y_pred = mnb.predict(x_test_resample)
print('\n*****************************************************')
print('\t\t\tReport Testing Set Results')
print('*****************************************************')
print(classification_report(y_test_resample, y_pred))

# pickle files
filename = 'vectorizer.pkl'
pickle.dump(vectorizer, open(filename, 'wb'))

filename = 'prediction.pkl'
pickle.dump(mnb, open(filename, 'wb'))

canv.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
window.mainloop()

