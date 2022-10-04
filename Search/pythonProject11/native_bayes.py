# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os

import socks
import socket
import requests
from bs4 import BeautifulSoup
import csv
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np
import pickle


def write_to_file(name_of_file, content):
    my_file = open(name_of_file,"w+")
    my_file.write(content)
    my_file.close()


def read_csv(name_of_csv_file):
    results = []
    with open('test2.csv') as File:
        reader = csv.DictReader(File)
        for row in reader:
            results.append(row)
        print(results)
    return results


def cleaning_text_from_tags(content_for_clean):
    soup = BeautifulSoup(content_for_clean,features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


def cleaning_punction_mark(content_for_clean):
    opt = re.sub(r'[^\w\s]', '', content_for_clean)
    return opt


def cleaning_stop_words(content_for_clean):
    stop_words_en = stopwords.words('english')
    querywords = content_for_clean.split()
    resultwords_en = [word for word in querywords if word.lower() not in stop_words_en]
    stop_words_ru = stopwords.words("russian")
    resultwords = [word for word in resultwords_en if word.lower() not in stop_words_ru]
    result = ' '.join(resultwords)
    return result


def cleaning_numbers(content_for_clean):
    opt = re.sub(r'\b\d+(?:\.\d+)?\s+', '', content_for_clean)
    return opt


def getaddrinfo(*args):
    return [(socket.AF_INET,socket.SOCK_STREAM,6,'',(args[0],args[1]))]


def get_link_from_site(cur_link):
    socks.set_default_proxy(socks.SOCKS5, "localhost", 9150)
    socket.socket = socks.socksocket
    socket.getaddrinfo = getaddrinfo
    res = requests.get(cur_link,verify=False)
    soup = BeautifulSoup(res.content, 'html.parser')
    soup.title
    links = [link.get('href') for link in soup.find_all('a')]
    links = list(filter(None, links))
    clean = []
    for l in links:
        if 'http' in l:
            clean.append(l)
    print(clean)
    return clean

def write_to_csv(name_of_csv,url_site,text,marker="none"):
    with open(name_of_csv, 'a+') as csvfile:
        fieldnames = ['index', 'link', 'text', 'marker']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if (text):
            writer.writerow({'index': cur_index, 'link': url_site, 'text': text, 'marker': marker})


def get_html_content(cur_link):
    socks.set_default_proxy(socks.SOCKS5, "localhost", 9150)
    socket.socket = socks.socksocket
    socket.getaddrinfo = getaddrinfo
    try:
        res = requests.get(cur_link,timeout=25)
    except:
        error = "ERROR"
        print(error,":", cur_link)
        return error
    print("Yes : ",cur_link)
    print(res)
    return res

cur_index =0

def create_my_data_set(bs_link, name_of_csv,marker,flag):
   links=get_link_from_site(bs_link)
   print(links)
   position = 0
   global cur_index
   with open(name_of_csv, 'a+') as csvfile:
        fieldnames = ['index', 'link', 'text', 'marker']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if(flag==0):
            writer.writeheader()
            cur_index = 0
        while position < len(links) - 1: 
            web_content = get_html_content(links[position])
            print(position)
            if(web_content!="ERROR"):
                clean_from_tags = cleaning_text_from_tags(web_content.content)
                clean_from_numbers = cleaning_numbers(clean_from_tags)
                clean_from_mark = cleaning_punction_mark(clean_from_numbers)
                clean = cleaning_stop_words(clean_from_mark)
                if(clean):
                    writer.writerow({'index' : cur_index,'link': links[position], 'text': clean, 'marker': marker})
                    cur_index = cur_index+1
            position = position + 1


def delete_duplicate(file_for_clean,file_clean):
    with open(file_for_clean, 'r') as infile, open(file_clean, 'w') as outfile:
        text_ns = []
        cur_id = 0
        results = csv.DictReader(infile)
        fieldnames = ['index', 'link', 'text', 'marker']
        writer = csv.DictWriter(outfile,fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            text_n = result.get('text')
            if text_n in text_ns:
                continue
            writer.writerow({'index':cur_id,'link':result.get('link'),'text':result.get('text'),'marker':result.get('marker')})
            cur_id = cur_id+1
            text_ns.append(text_n)


def association_data_set(first,second,result):
    with open(first, 'r') as infile1, open(second, 'r') as infile2, open(result, "w") as outfile:
        part_1 = csv.DictReader(infile1)
        part_2 = csv.DictReader(infile2)
        id=0
        fieldnames = ['index', 'link', 'text', 'marker']
        writer = csv.DictWriter(outfile,fieldnames=fieldnames)
        writer.writeheader()
        for row_1 in part_1:
            writer.writerow({'index': id,'link': row_1.get('link'),'text': row_1.get('text'),'marker': row_1.get('marker')})
            id = id + 1
        for row_2 in part_2:
            writer.writerow({'index': id, 'link': row_2.get('link'), 'text': row_2.get('text'), 'marker': row_2.get('marker')})
            id = id + 1


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def make_model(counts, df):
    X_train, X_test, y_train , y_test = train_test_split(counts, df['marker'], test_size=0.1, random_state=99)
    model = MultinomialNB().fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(np.mean(predicted == y_test))
    print(confusion_matrix(y_test, predicted))
    preds = predicted
    fpr, tpr, threslhold = roc_curve(y_test,preds)
    roc_auc = auc(fpr,tpr)
    import matplotlib.pyplot as plt
    plt.title('Receiver Operation Characteristic')
    plt.plot(fpr,tpr,'b',label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1],[0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    save_model(model)

filename = 'classifier/finalized_model.pkl'

def special_model(name_of_file):
    df = pd.read_csv(name_of_file)
    df['marker'] = df.marker.map({'good': 0, 'bad': 1, 'none': 2})
    print(df['marker'])
    df['text'] = df.text.astype(str)
    df['text'] = df.text.map(lambda x: x.lower())
    print(df['text'])
    nltk.download('punkt')
    df['text'] = df['text'].apply(nltk.word_tokenize)
    print(df['text'])
    stemmer = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: [stemmer.stem(y) for y in x])
    df['text'] = df['text'].apply(lambda x: ' '.join(x))
    print(df['text'])

    with open(filename, 'rb') as f:
        count_vect, transformer = pickle.load(f)

    counts = count_vect.transform(df['text'])
    counts = transformer.transform(counts)

    model = load_model()
    predicted = model.predict(counts)
    print("Result: ", predicted)
    return predicted





def data_preparation(name_of_file):
    df = pd.read_csv(name_of_file)
    df['marker'] = df.marker.map({'good': 0, 'bad': 1, 'none': 2})
    print(df['marker'])
    df['text'] = df.text.astype(str)
    df['text'] = df.text.map(lambda x: x.lower())
    print(df['text'])
    nltk.download('punkt')
    df['text'] = df['text'].apply(nltk.word_tokenize)
    print(df['text'])
    stemmer = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: [stemmer.stem(y) for y in x])
    df['text'] = df['text'].apply(lambda x: ' '.join(x))
    print(df['text'])
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(df['text'])
    transformer = TfidfTransformer().fit(counts)
    counts = transformer.transform(counts)
    print(counts)
    save_all(count_vect,transformer)
    make_model(counts, df)




def save_all(count_vect, transformer):
    with open(filename, 'wb') as fout:
        pickle.dump((count_vect, transformer), fout)


joblib_file = "classifier/joblib_model.pkl"

def save_model(model):
    joblib.dump(model, joblib_file)
    print("Model saved successfully")

def load_model():
    joblib_model = joblib.load(joblib_file)
    return joblib_model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
  special_model("classifier/gotov.csv")













# See PyCharm help at https://www.jetbrains.com/help/pycharm/
