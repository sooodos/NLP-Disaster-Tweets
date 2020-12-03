"""
    Description: In this file we try to optimize the model that yields the
    best f1-score for our given dataset which is the Logistic Regression model.
    We do this by trying various ngrams for both count vectorizer and tfidf and
    choosing which one is the best and also inspect the possibility for feature
    reduction.

    Authors: Marcos Antonios Charalambous (mchara01@cs.ucy.ac.cy)
             Sotiris Loizidis (sloizi02@cs.ucy.ac.cy)

    Date: 26/04/2020
"""
import sys  # For system calls such us std.out
import pandas as pd  # CSV file I/O (e.g. pd.read_csv).
import warnings  # Omit any warnings in output.

import numpy as np  # Linear algebra.
import matplotlib.pyplot as plt  # Used in graph creation

import re  # For regular expression operations.
import unicodedata  # Character properties for all unicode characters.
import string  # Tools to manipulate strings.
import html
import twokenize
import spacy  # Advanced operations for natural language processing.
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from stop_words import safe_get_stop_words
from bs4 import BeautifulSoup  # For decoding HTML to general text.

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from ekphrasis.classes.segmenter import Segmenter  # ekphrasis library segmenter for tweets

from abbreviation import abbreviations  # various abbreviations collected

# NEW MODULES
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.pipeline import Pipeline
from time import time

sys.stdout = open("./output/ngrams_fselection.txt", "w")

warnings.filterwarnings("ignore")

plt.style.use('ggplot')

nlp = spacy.load('en_core_web_sm')
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False

lemmatizer = WordNetLemmatizer()
stop_words = safe_get_stop_words('en')
hashtag_regex = re.compile(r"\#\b[\w\-\_]+\b")
twitter_segmenter = Segmenter(corpus="twitter_2018")
camelcase_regex = re.compile(r'((?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])|[0-9]+|(?<=[0-9\-\_])[A-Za-z]|[\-\_])')


# DATA PRE-PROCESSING FUNCTIONS
def unescape_tweet(tweet):
    """Unescaping various chars found in text """
    return html.unescape(tweet)


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, 'lxml')
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def convert_accented_chars(text):
    """Convert accented characters from text, e.g. café"""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text):
    """Contractions are shortened version of words or syllables.
        Converting each contraction to its expanded, original form
        helps with text standardization."""
    text = re.sub(
        r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't",
        r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)

    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

    return text


def replace_special(text):
    """Convert all special characters found in text with
       characters we can work with"""
    text = text.replace('\r\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('``', "''")
    text = text.replace('`', "'")
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace("'", "'")
    text = text.replace('–', "-")
    text = text.replace('\"', '"')
    text = text.replace("\'", "'")
    return text


def expand_hashtag(match):
    """Expand hashtags found in tweets using ekphrasis library"""
    hashtag = match.group()[1:]

    if hashtag.islower():
        expanded = twitter_segmenter.segment(hashtag)
        expanded = " ".join(expanded.split("-"))
        expanded = " ".join(expanded.split("_"))
    else:
        expanded = camelcase_regex.sub(r' \1', hashtag)
        expanded = expanded.replace("-", "")
        expanded = expanded.replace("_", "")
    return "#" + hashtag + " " + expanded


def expand_tweet(tweet):
    """Expand hashtags found in tweets using ekphrasis library"""
    return hashtag_regex.sub(lambda hashtag: expand_hashtag(hashtag), tweet)


def remove_mentions(text):
    """Remove all mentions which start with '@' follow by any non-whitespace character"""
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'@', "at", text)
    return text


def remove_url(text):
    """Here we deal with the URLs which we replace with empty strings"""
    url = re.sub('https?://[A-Za-z0-9./]+', '', text)
    return url


def convert_abbrev(word):
    """Helper method for convert_abbrev_in_text"""
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word


def convert_abbrev_in_text(text):
    """Convert all abbreviations found in given text. Abbreviations are
        in file abbreviation.py"""
    tokens = word_tokenize(text)
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text


def remove_emoji(text):
    """Remove all available emojis from a given text"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_stop_words(text):
    """Remove stop words"""
    twokens = twokenize.tokenizeRawTweetText(text)
    twokens = [t for t in twokens if t.lower() not in stop_words]
    return ' '.join(twokens)


def lemmatize(text):
    """Lemmatize words"""
    twokens = twokenize.tokenizeRawTweetText(text)
    twokens = [lemmatizer.lemmatize(twoken) for twoken in twokens]
    return ' '.join(twokens)


def remove_punct(text):
    """Remove any remaining punctuations"""
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def reduce_spaces(tweet):
    """Remove extra whitespaces from text"""
    text = tweet.strip()
    text = " ".join(text.split())
    return re.sub(' +', ' ', text)


def tweet_cleaner(text, unescape=True, remove_html_tags=True, accented_chars=True,
                  contractions=True, special_chars=True, expand_hash=True, remove_mention=True,
                  remove_links=True, convert_abbrevations=True, remove_all_emojis=True,
                  remove_stop=False, remove_num=True, lemmatization=True, lowercase=True):
    """Preprocess text with default option set to true for all steps. Stop words are kept because they
        can cause a drop in performance otherwise."""
    if unescape:  # unescape tweets
        unescape_tweet(text)
    if remove_html_tags:  # remove html tags
        text = strip_html_tags(text)
    if accented_chars:  # remove accented characters
        text = convert_accented_chars(text)
    if contractions:  # expand contractions
        text = expand_contractions(text)
    if special_chars:  # convert any special characters
        text = replace_special(text)
    if expand_hash:  # expand words in hashtags
        text = expand_tweet(text)
    if remove_mention:  # remove twitter mentions which start with @ and hashtags
        text = remove_mentions(text)
    if remove_links:  # remove all links in a tweet which start with http or https
        text = remove_url(text)
    if convert_abbrevations:  # convert all abbreviations found to their normal form
        text = convert_abbrev_in_text(text)
    if remove_all_emojis:  # remove all emojis from given text
        text = remove_emoji(text)
    if remove_stop:  # remove stop words
        text = remove_stop_words(text)
    if lemmatization:  # convert tokens to base form
        text = lemmatize(text)
    if lowercase:
        text = text.lower()

    text = remove_punct(text)
    text = reduce_spaces(text)

    doc = nlp(text)  # tokenize text

    clean_text = []

    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words and token.is_stop and token.pos_ != 'NUM':
            flag = False
        # remove all numbers
        if remove_num and (token.pos_ == 'NUM' or token.text.isnumeric()) and flag:
            flag = False
        # convert tokens to base form
        if lemmatization and token.lemma_ != "-PRON-" and flag:
            edit = token.lemma_
        # append tokens edited and not removed to list
        if edit != "" and flag:
            clean_text.append(edit)
    return (" ".join(clean_text)).strip()


# END OF DATA PRE-PROCESSING METHODS


# Read our train dataset
train_df = pd.read_csv("./train.csv")

print("Cleaning Tweets...")
ti = time()
train_df["text"] = train_df["text"].apply(lambda s: tweet_cleaner(s))
preprocessing_phase_time = time() - ti
print("End of data preprocessing phase. Time taken: {0:.2f}s".format(preprocessing_phase_time))

X = train_df["text"]
Y = train_df["target"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)


def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test) * 1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test) * 1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test) * 1.))
    disaster_fit = pipeline.fit(x_train, y_train)
    y_pred = disaster_fit.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("null accuracy: {0:.2f}%".format(null_accuracy * 100))
    print("accuracy score: {0:.2f}%".format(accuracy * 100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy - null_accuracy) * 100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy - accuracy) * 100))
    print("-" * 80)
    return accuracy


cvec = CountVectorizer()
tvec = TfidfVectorizer()
lr = LogisticRegression()  # Model that yields best results for our data
n_features = np.arange(1000, 10000, 1000)


def nfeature_accuracy_checker(vectorizer=cvec, stop_words=None, n_features=n_features, ngram_range=(1, 1),
                              classifier=lr):
    result = []
    print(classifier)
    print()
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Validation result for {} features".format(n))
        nfeature_accuracy = accuracy_summary(checker_pipeline, x_train, y_train, x_test, y_test)
        result.append((n, nfeature_accuracy))
    return result


# Count Vectorizer
feature_result_ug = nfeature_accuracy_checker(ngram_range=(1, 1))
feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))
feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))

nfeatures_plot_tg = pd.DataFrame(feature_result_tg, columns=['nfeatures', 'validation_accuracy'])
nfeatures_plot_bg = pd.DataFrame(feature_result_bg, columns=['nfeatures', 'validation_accuracy'])
nfeatures_plot_ug = pd.DataFrame(feature_result_ug, columns=['nfeatures', 'validation_accuracy'])

# TF-IDF
feature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec, ngram_range=(1, 1))
feature_result_bgt = nfeature_accuracy_checker(vectorizer=tvec, ngram_range=(1, 2))
feature_result_tgt = nfeature_accuracy_checker(vectorizer=tvec, ngram_range=(1, 3))

nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt, columns=['nfeatures', 'validation_accuracy'])
nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt, columns=['nfeatures', 'validation_accuracy'])
nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt, columns=['nfeatures', 'validation_accuracy'])

# Graph with n-grams from 1 to 3 for both count vectorizer and tf-idf
plt.figure(figsize=(8, 6))
plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy, label='Trigram TF-IDF',
         color='royalblue')  # Trigram using tf-idf
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy, label='Trigram Count Vectorizer',
         linestyle=':', color='royalblue')  # Trigram using count vectorizer
plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy, label='Bigram TF-IDF',
         color='orangered')  # Bigram using tf-idf
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy, label='Bigram Count Vectorizer',
         linestyle=':', color='orangered')  # Bigram using count vectorizer
plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='Unigram TF-IDF',
         color='gold')  # Unigram using tfidf
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='Unigram Count Vectorizer',
         linestyle=':', color='gold')  # Unigram using count vectorizer
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
plt.savefig('./output/ngrams_cvec_tfidf.png', bbox_inches='tight')
plt.show()

# Check different accuracy achieved for TF-IDF by using stop words or not in our data
feature_result_ug = nfeature_accuracy_checker(vectorizer=tvec)
feature_result_wosw = nfeature_accuracy_checker(vectorizer=tvec, stop_words='english')

nfeatures_plot_ug = pd.DataFrame(feature_result_ug, columns=['nfeatures', 'validation_accuracy'])
nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw,
                                      columns=['nfeatures', 'validation_accuracy'])

plt.figure(figsize=(8, 6))
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='with stop words')
plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy, label='without stop words')
plt.title("TF-IDF - Without stop words VS With stop words (Unigram): Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
plt.savefig('./output/stop_nostop_tfidf.png', bbox_inches='tight')
plt.show()

# FEATURE SELECTION USING CHI2
temp_tfidf = TfidfVectorizer(ngram_range=(1, 1))
temp = temp_tfidf.fit_transform(x_train)
max_f = temp.shape[1]  # Used only for extracting the dimension of a tfidf vector using unigrams

tvec = TfidfVectorizer(max_features=max_f, ngram_range=(1, 1))
x_train_tfidf = tvec.fit_transform(x_train)
x_validation_tfidf = tvec.transform(x_test)
chi2score = chi2(x_train_tfidf, y_train)[0]  # Calculate chi2 scores from our tfidf vector

# Plot the most useful unigram features selected by chi2 for predicting either one of the two classes
plt.figure(figsize=(15, 10))
wscores = list(zip(tvec.get_feature_names(), chi2score))
wchi2 = sorted(wscores, key=lambda x: x[1])
topchi2 = list(zip(*wchi2[-20:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x, topchi2[1], align='center', alpha=0.2)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.savefig('./output/topchi2.png', bbox_inches='tight')
plt.show()

# Reduce the dimensions to different numbers of features, and also check the accuracy on the validation set.
ch2_result = []
for n in np.arange(1000, 10000, 1000):
    ch2 = SelectKBest(chi2, k=n)
    x_train_chi2_selected = ch2.fit_transform(x_train_tfidf, y_train)
    x_validation_chi2_selected = ch2.transform(x_validation_tfidf)
    clf = LogisticRegression()
    clf.fit(x_train_chi2_selected, y_train)
    score = clf.score(x_validation_chi2_selected, y_test)
    ch2_result.append(score)
    print("chi2 feature selection evaluation calculated for {} features".format(n))

# Compare the validation accuracy at the same number of features when the number of features has been limited from
# Tfidf vectorizing stage and also when the number of features has been reduced from 7.500 features using chi2 statistic.
plt.figure(figsize=(8, 6))
plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='Unigram TF-IDF',
         color='royalblue')
plt.plot(np.arange(1000, 10000, 1000), ch2_result, label='TF-IDF dimensions reduced from 10000 features', linestyle=':',
         color='orangered')

plt.title("Features limited within tfidft vectorizer VS Reduced dimensions with chi2")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
plt.savefig('./output/tfidf_vs_chi2.png', bbox_inches='tight')
plt.show()

sys.stdout.close()
