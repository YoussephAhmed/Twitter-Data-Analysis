#!/usr/bin/env python
import tweepy, sys, os
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from termcolor import colored
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as stopwordsNLTK
import spacy
import nltk
import pandas as pd
import regex as re
import itertools
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import random
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from matplotlib import interactive
import warnings
warnings.filterwarnings("ignore")
import arabic_reshaper # this was missing in your code
from bidi.algorithm import get_display
import tkinter as tk
import tkinter.scrolledtext as tkscrolled
from tkinter import *


from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure







def plotthem(data1,data2):
    plt.figure(1)
    plt.imshow(data1, interpolation='bilinear')
    plt.axis("off")
    plt.gcf().canvas.draw()


    positive = data2[0]
    negative = data2[1]

    labels = ['Positive [' + str(positive) + '%]', 'Negative [' + str(negative) + '%]']
    sizes = [positive, negative]
    colors = ['yellowgreen', 'red']

    plt.figure(2)
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.title(" by analyzing " + str(noOfSearchTerms) + " Tweets.")
    plt.axis('equal')
    plt.tight_layout()
    plt.gcf().canvas.draw()

    '''
    plt.figure(2)
    plt.clf()
    x = np.arange(0.0,3.0,0.01)
    y = np.tan(2*np.pi*x+random.random())
    plt.plot(x,y)
    plt.gcf().canvas.draw()

    '''














# plot function is created for
# plotting the graph in
# tkinter window
def plot(data1):
    # the figure that will contain the plot

    fig = plt.figure(1)
    #fig = plt.subplot(2, 1, 1)
    plt.imshow(data1, interpolation='bilinear')
    plt.axis("off")

    # adding the subplot
    #plot1 = fig.add_subplot(111)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master=root)


    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()




def percentage(part, whole):
    return 100 * float(part) / float(whole)

##################### MACHINE LEARNING PART

def read_tsv(data_file):
    text_data = list()
    labels = list()
    infile = open(data_file, encoding='utf-8')
    for line in infile:
        if not line.strip():
            continue
        label, text = line.split('\t')
        text_data.append(text)
        labels.append(label)
    return text_data, labels


def classify(input_text, classifier, x_train, y_train):

    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=0.0001, max_df=0.95,
                                 analyzer='word', lowercase=False,
                                 )),
        ('clf', classifier),
    ])

    pipeline.fit(x_train, y_train)
    feature_names = pipeline.named_steps['vect'].get_feature_names()

    y_predicted = pipeline.predict(input_text)

    return y_predicted, feature_names

def load(pos_train_file, neg_train_file, pos_test_file, neg_test_file):
    pos_train_data, pos_train_labels = read_tsv(pos_train_file)
    neg_train_data, neg_train_labels = read_tsv(neg_train_file)

    pos_test_data, pos_test_labels = read_tsv(pos_test_file)
    neg_test_data, neg_test_labels = read_tsv(neg_test_file)

    x_train = pos_train_data + neg_train_data
    y_train = pos_train_labels + neg_train_labels

    x_test = pos_test_data + neg_test_data
    y_test = pos_test_labels + neg_test_labels

    return x_train, y_train, x_test, y_test

##################### DATA ANALYTICS PART


def show_home_tweets_to_txt(api,n):
    for status in tweepy.Cursor(api.home_timeline).items(n):
        print(status._json)



def show_world_locations(api):
    # Get all the locations where Twitter provides trends service
    worldTrendsLocations = api.trends_available()
    print(worldTrendsLocations)


# country is constant per region
# links for all the region:
# world wide id = 1
# Egypt id = 28584965
def trends_for_region_to_txt(region_id):
    f = open("trendings.txt", "w+", encoding="utf-8")
    trends = api.trends_place(id=region_id)

    width, height = 50, 20
    TKScrollTXT = tkscrolled.ScrolledText(root, width=width, height=height, wrap='word')
    TKScrollTXT.pack()


    for value in trends:
        for trend in value['trends']:
            f.write("This is a trend -> " + trend['name'] + "\n")
            f.write("-------------------- \n")

            TKScrollTXT.insert(INSERT, trend['name'] + "\n")

    TKScrollTXT.config(state=DISABLED)

    f.close()


# the past few weeks of the tweets
def tweets_of_topic_to_txt(api,searchTerm,date,noOfSearchTerms=100, lang="ar"):
    tweets = api.search(q=searchTerm, date=date, lang=lang, count=noOfSearchTerms, tweet_mode="extended")
    return tweets


def tweeters_of_topic_to_txt(topic_tweets):
    f = open("tweeters.txt", "w+", encoding="utf-8")
    users_locs = [[tweet.user.screen_name, tweet.user.location] for tweet in topic_tweets]

    for value in users_locs:
        f.write("The user name: " + value[0] + " -. [region]: "  + value[1]  + "\n")
        f.write("-------------------- \n")

    f.close()

def tweets_to_txt(tweets):
    f = open("tweets.txt", "w+", encoding="utf-8")
    for tweet in tweets:
        tweet_line = tweet.full_text.replace("\n"," ")
        f.write( tweet_line + "\n")
        #f.write("-------------------- \n")

    f.close()

def word_freq(tweets):
    f = open("word_freq.txt", "w+", encoding="utf-8")
    words_in_tweet = [tweet.full_text.split() for tweet in tweets]
    words_in_tweet_strings = " "

    for tweet in tweets:
        words_in_tweet_strings += tweet.full_text

    # List of all words across tweets
    all_words_all_tweets = list(itertools.chain(*words_in_tweet))
    # Create counter
    counts = collections.Counter(all_words_all_tweets)

    for value in counts.most_common(20):
        f.write("The word '" + value[0] + "' is repeated -> " + str(value[1]) + "\n")
        f.write("-------------------- \n")
    f.close()


def visualization(tweets,y_predicted,noOfSearchTerms):
    # Create and generate a word cloud image:
    words_in_tweet_strings = " "

    weridPatterns = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              u"\U00002702-\U000027B0"
                              u"\U000024C2-\U0001F251"
                              u"\U0001f926-\U0001f937"
                              u'\U00010000-\U0010ffff'
                              u"\u200d"
                              u"\u2640-\u2642"
                              u"\u2600-\u2B55"
                              u"\u23cf"
                              u"\u23e9"
                              u"\u231a"
                              u"\u3030"
                              u"\ufe0f"
                              u"\u2069"
                              u"\u2066"
                              u"\u200c"
                              u"\u2068"
                              u"\u2067"
                              "]+", flags=re.UNICODE)

    for tweet in tweets:
        words_in_tweet_strings += tweet.full_text

    clean_text = weridPatterns.sub(r'', words_in_tweet_strings)

    data = arabic_reshaper.reshape(clean_text)
    data = get_display(data)  # add this line
    wordcloud = WordCloud(font_path='arial', background_color='white',
                          mode='RGB', width=2000, height=1000).generate(data)


    ####### Visualization
    n_all = len(y_predicted)
    n_pos = 0
    n_neg = 0

    for y in y_predicted:
        if y == 'pos':
            n_pos += 1

        if y == 'neg':
            n_neg += 1

    positive = percentage(n_pos, n_all)
    negative = percentage(n_neg, n_all)

    positive = format(positive,'.2f')
    negative = format(negative, '.2f')

    plotthem(wordcloud,[positive,negative])


##############

intro = '''
___ _ _ _ _ ___ ___ ____ ____    
 |  | | | |  |   |  |___ |__/    
 |  |_|_| |  |   |  |___ |  \ 
'''

print(colored(intro, 'blue'))


consumerKey = "T2jXUHLe4dSvtlmZYuoZH6yeA"
consumerSecret = "jetuh1jZaXGTHrZ0hFMkAGm695SRc6EzX6Rw6dyUqsdnuPlzYC"
accessToken = "1285916588697362435-liwPeMLnktBFrzwnIX7IPlekJhPj8f"
accessTokenSecret = "tsyyuNVO82bzqx5WAsBdo8es0WWtXCXuWOZtzo11f23HA"


auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)

searchTerm =""
noOfSearchTerms = 10
#searchTerm = input("Enter keyword/hashtag to search about: ")
#noOfSearchTerms = int(input("Enter how many tweets to analyze: "))
#searchTerm = searchTerm + " -filter:retweets"
date_since = "2020-8-15"








def analyze():
    # init figures
    fig1 = plt.figure()
    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.get_tk_widget().pack()

    fig2 = plt.figure()
    canvas2 = FigureCanvasTkAgg(fig2, master=root)
    canvas2.get_tk_widget().pack()

    # timeline
    # show_home_tweets_to_txt(api,10)

    # tweets of a given hashtag
    tweets = tweets_of_topic_to_txt(api, searchTerm, date_since)
    tweets_to_txt(tweets)

    # tweeters of certain topic
    tweeters_of_topic_to_txt(tweets)

    # word freq
    word_freq(tweets)

    # visualization

    pos_training = 'input/train_Arabic_tweets_positive_20190413.tsv'
    neg_training = 'input/train_Arabic_tweets_negative_20190413.tsv'

    pos_testing = 'input/test_Arabic_tweets_positive_20190413.tsv'
    neg_testing = 'input/test_Arabic_tweets_negative_20190413.tsv'

    # sample tweets from text file now
    with open('tweets.txt', encoding="utf8") as f:
        input_tweets = [line.rstrip() for line in f]

    # available classifiers:
    '''
    LinearSVC(), SVC(), MultinomialNB(),
                   BernoulliNB(), SGDClassifier(), DecisionTreeClassifier(max_depth=5),
                   RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                   KNeighborsClassifier(3)
    '''

    width, height = 70, 20
    out = tkscrolled.ScrolledText(root, width=width, height=height, wrap='word')
    out.place(x=40, y=570)


    f = open("tweets_classified.txt", "w+", encoding="utf-8")
    x_train, y_train, x_test, y_test = load(pos_training, neg_training, pos_testing, neg_testing)
    y_predicted, feature_names = classify(input_tweets, BernoulliNB(), x_train, y_train)
    for i in range(len(y_predicted)):
        f.write(str(i) + "-Tweet:  " + input_tweets[i] + "   --->  " + y_predicted[i] + "\n")
        f.write("-------------------- \n")

        out.insert(INSERT, input_tweets[i] + "   --->  " + y_predicted[i] + "\n")
        out.insert(INSERT, "=====================================" + "\n")

    out.config(state=DISABLED)
    out.xview_moveto(1)
    f.close()


    visualization(tweets, y_predicted, noOfSearchTerms)


def getData():
    global searchTerm
    global noOfSearchTerms
    x1 = entry1.get()
    x2 = entry2.get()
    searchTerm = x1
    noOfSearchTerms = int(x2)
    analyze()


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



root = tk.Tk()
root.title('TWITTER ANALYSIS')
root.geometry("2000x1000") #Width x Height



wt =Label(root,text="Egypt trends")
wt.pack()

# show hashtags for a given region
# show_world_locations(api)
trends_for_region_to_txt(23424802)  # woeid for Egypt


canvas1 = tk.Canvas(root, width=400, height=100)
canvas1.pack()




keyword = Label(root,
                  text="keyword").place(x = 600,
                                           y = 375)



number = Label(root,
                      text="number").place(x = 600,
                                               y = 405)



entry1 = tk.Entry(root)
canvas1.create_window(200, 40, window=entry1)

entry2 = tk.Entry(root)
canvas1.create_window(200, 70, window=entry2)

button1 = tk.Button(text='Analyze', command=getData)
canvas1.create_window(200, 95, window=button1)
searchTerm = searchTerm + " -filter:retweets"
#date_since = "2020-8-15"


output_text = Label(root,
                  text="output tweets:").place(x = 40,
                                           y = 540)


root.mainloop()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

