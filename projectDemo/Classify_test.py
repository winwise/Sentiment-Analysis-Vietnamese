from sklearn.datasets import load_files
import numpy as np
import re
from underthesea import sent_tokenize, word_tokenize
from collections import Counter
from string import punctuation

corpus = load_files("Data",encoding="utf-8",load_content=True)
X, y = corpus.data, corpus.target
# Buoc 1: tien xu ly van ban
documents = []
# + Viet ham xoa cac the HTML va bieu tuong cam xuc
def cleanHTML(text):
    cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});<.*?>')
    cleantext = re.sub(cleaner,'',text)
    return cleantext
def delEmoji(text):    
    emoji_pattern = re.compile("["
     u"\U0001F600-\U0001F64F" 
     u"\U0001F300-\U0001F5FF" 
     u"\U0001F680-\U0001F6FF" 
     u"\U0001F1E0-\U0001F1FF"
     u"\U00002500-\U00002BEF"
     u"\U00002702-\U000027B0"
     u"\U00002702-\U000027B0"
     u"\U000024C2-\U0001F251"
     u"\U0001f926-\U0001f937"
     u"\U00010000-\U0010ffff"
     u"\u2640-\u2642"
     u"\u2600-\u2B55"
     u"\u200d"
     u"\u23cf"
     u"\u23e9"
     u"\u231a"
     u"\ufe0f"
     u"\u3030"
     u"\xa0"
     "]+", flags=re.UNICODE)
    txt = re.sub(emoji_pattern,'',text)
    txt = txt.replace("\n"," ")
    return txt
# Tien xu ly lan thu nhat, xoa het tat ca cac the HTML va Emoji trong X
def tienXuLyLan1(dcmts):
    temp = []
    for cmt in dcmts:
        cmt = cleanHTML(cmt)
        cmt = delEmoji(cmt)
        temp.append(cmt)
    return temp
documents = tienXuLyLan1(X)
# Tien xu ly lan thu hai, loai bo stop words va dau cau, gom nhom cac tu
def Vistopwords():
    stop_word = []
    with open("Vistopwords.txt",'r+',encoding="utf-8") as f_read:
        text = f_read.read()
        for word in text.split(" "):
            stop_word.append(word)
        f_read.close()
    punc = list(punctuation)
    stop_word = stop_word + punc
    return stop_word
def tienXuLyLan2(dcmts):
    sentences = []
    for cmt in dcmts:
        word_cmt = word_tokenize(cmt) # word_cmt la mot list cac tu
        sent = ""
        for word in word_cmt:
            if word not in Vistopwords():
                sent = sent + word + " "
        sentences.append(sent)
    return sentences
documents = tienXuLyLan2(documents)
# Buoc 3: Chuyen doi Text thanh Numbers dung The Bag of Words Model
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(min_df=5,max_df= 0.8,max_features=2000,sublinear_tf=True)
X = tf.fit_transform(documents).toarray()
# Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Script phia tren chia du lieu thanh 2 phan 80% tap train va 20% tap test
# Buoc 5: Su dung LogisticRegression Algorithm de train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# Thu nghiem ty cho vui thoi
txt = ["quán nấu ăn ngon lắm, hi vọng lần sau sẽ đến nữa","dưới chất lượng cho phép","không hợp vệ sinh"]
txt = tienXuLyLan1(txt)
txt = tienXuLyLan2(txt)
print(txt)
test_txt = tf.transform(txt)
print(model.predict(test_txt))