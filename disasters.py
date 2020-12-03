"""
    Description: This file constitutes the center of our project.
    It contains methods for exploratory data analysis of our
    dataset, various data pre-processing functions specialized for
    tweets, detecting the best word embedding to transform the
    words in our tweets in vectors and various kinds of models to
    find which one yields the best result for our process data.

    Authors: Marcos Antonios Charalambous (mchara01@cs.ucy.ac.cy)
             Sotiris Loizidis (sloizi02@cs.ucy.ac.cy)

    Date: 26/04/2020
"""
import sys  # For system calls such us std.out
import pandas as pd  # CSV file I/O (e.g. pd.read_csv).
import warnings  # Omit any warnings in output.

import numpy as np  # Linear algebra.
import matplotlib
import matplotlib.pyplot as plt  # Used in graph creation
import matplotlib.patches as mpatches  # Used in graph creation
import seaborn as sns  # Useful for visualizing correlation

import gensim.downloader as api  # Downloading twitter pretrained vectores

import re  # For regular expression operations.
import unicodedata  # Character properties for all unicode characters.
import string  # Tools to manipulate strings.
import html  # For html manipulation (unescape)
import nltk
import twokenize  # Twitter tokenizer
import spacy  # Advanced operations for natural language processing.
from nltk.stem import WordNetLemmatizer  # Lemmatizer for tweet processing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from stop_words import safe_get_stop_words
from collections import defaultdict
from collections import Counter
from bs4 import BeautifulSoup  # For decoding HTML to general text.

# All sklearn modules imported are shown belown
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import scale

from wordcloud import WordCloud  # Showing frequent word post data processing

from ekphrasis.classes.segmenter import Segmenter  # ekphrasis module segmenter for tweets

from time import time  # Printing time taken for methods

from abbreviation import abbreviations  # various abbreviations collected
# from contractions_map import CONTRACTION_MAP  # Can be uncomment and used instead if desired.

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

warnings.filterwarnings("ignore")

sys.stdout = open("./output/disaster_output.txt", "w")

plt.style.use('ggplot')

nlp = spacy.load('en_core_web_sm')
deselect_stop_words = ['no', 'not']  # we don't consider no and not stop words
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
    return "#" + hashtag + " " + expanded  # returns the hashtag and its expanded form


def expand_tweet(tweet):
    """Expand hashtags found in tweets using ekphrasis library"""
    return hashtag_regex.sub(lambda hashtag: expand_hashtag(hashtag), tweet)


def remove_mentions(text):
    """Remove all mentions which start with '@' follow by any
       non-whitespace character"""
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
    if lowercase:  # convert all text to lowercase
        text = text.lower()
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


def cv(data):
    """Count Vectorizer initializer"""
    count_vec = CountVectorizer()

    emb = count_vec.fit_transform(data)

    return emb, count_vec


def tfidf(data):
    """TF/IDF initializer"""
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer


def get_metrics(y_test, y_predicted):
    """Calculation of various metrics regarding natural language processing
       This metrics help us to see if our classifier performed well"""
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


# Read our train dataset
train_df = pd.read_csv("./train.csv")

### EXPLORATORY DATA ANALYSIS (EDA) OF TWEETS ###
# Some basic EDA first to inspect our dataset before we try various ways to process it.

print("Number of samples: {}".format(len(train_df)))
print('There are {} rows and {} columns in train dataset'.format(train_df.shape[0], train_df.shape[1]))
print("Class label distribution")
print("-Number of positive samples: {}".format(len(train_df.loc[train_df['target'] == 1])))
print("-Number of negative samples: {}".format(len(train_df.loc[train_df['target'] == 0])))

# Extracting the number of examples of each class
disaster_len = train_df[train_df['target'] == 1].shape[0]
not_disaster_len = train_df[train_df['target'] == 0].shape[0]

# bar plot of the 2 classes
plt.rcParams['figure.figsize'] = (7, 5)
plt.bar(10, disaster_len, 3, label="Disaster", color='red')
plt.bar(15, not_disaster_len, 3, label="Not Disaster", color='green')
plt.legend()
plt.ylabel('Number of tweets')
plt.title('Tweets class distribution')
plt.gca().axes.get_xaxis().set_visible(False)
plt.savefig('./output/two_classes.png', bbox_inches='tight')
plt.show()

# Basic analysis on character level, word level and sentence level.

# Number of characters in tweets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
tweet_len = train_df[train_df['target'] == 1]['text'].str.len()
ax1.hist(tweet_len, color='red')
ax1.set_title('Disaster tweets')
tweet_len = train_df[train_df['target'] == 0]['text'].str.len()
ax2.hist(tweet_len, color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.savefig('./output/chars_tweets.png', bbox_inches='tight')
plt.show()

# Number of words in a tweet
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
tweet_len = train_df[train_df['target'] == 1]['text'].str.split().map(lambda s: len(s))
ax1.hist(tweet_len, color='red')
ax1.set_title('Disaster tweets')
tweet_len = train_df[train_df['target'] == 0]['text'].str.split().map(lambda s: len(s))
ax2.hist(tweet_len, color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Words in a tweet')
plt.savefig('./output/words_tweets.png', bbox_inches='tight')
plt.show()

# Average word length in a tweet
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
word = train_df[train_df['target'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)), ax=ax1, color='red')
ax1.set_title('Disaster')
word = train_df[train_df['target'] == 0]['text'].str.split().apply(lambda x: [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)), ax=ax2, color='green')
ax2.set_title('Not disaster')
fig.suptitle('Average word length in each tweet')
plt.savefig('./output/word_length_tweet.png', bbox_inches='tight')
plt.show()


def create_corpus(target):
    corpus_lst = []

    for s in train_df[train_df['target'] == target]['text'].str.split():
        for i in s:
            corpus_lst.append(i)
    return corpus_lst


stop = set(stopwords.words('english'))

# Analyze the stopwords of tweets with class 1.

corpus = create_corpus(1)

dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word] += 1

top = sorted(dic.items(), key=lambda s: s[1], reverse=True)[:10]

plt.rcParams['figure.figsize'] = (18.0, 6.0)
x, y = zip(*top)
plt.bar(x, y, color='red')
plt.title("Common stopwords in disaster tweets")
plt.savefig('./output/stop_disaster.png', bbox_inches='tight')
plt.show()

# Analyze the stopwords of tweets with class 0.
corpus = create_corpus(0)

dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word] += 1

top = sorted(dic.items(), key=lambda s: s[1], reverse=True)[:10]

plt.rcParams['figure.figsize'] = (18.0, 6.0)
x, y = zip(*top)
plt.bar(x, y, color='green')
plt.title("Common stopwords in non-disaster tweets")
plt.savefig('./output/stop_irrelevant.png', bbox_inches='tight')
plt.show()

# Analyzing punctuations of tweets indicating a real disaster.
plt.figure(figsize=(10, 5))
corpus = create_corpus(1)

dic = defaultdict(int)

special = string.punctuation
for i in corpus:
    if i in special:
        dic[i] += 1

x, y = zip(*dic.items())
plt.bar(x, y, color='red')
plt.title("Punctuations in disaster tweets")
plt.savefig('./output/punc_disaster.png', bbox_inches='tight')
plt.show()

# Analyzing punctuations of non-disaster tweets.
plt.figure(figsize=(10, 5))
corpus = create_corpus(0)

dic = defaultdict(int)

special = string.punctuation
for i in corpus:
    if i in special:
        dic[i] += 1

x, y = zip(*dic.items())
plt.bar(x, y, color='green')
plt.title("Punctuations in non-disaster tweets")
plt.savefig('./output/punc_irrelevant.png', bbox_inches='tight')
plt.show()

# Common words in all tweets.
plt.figure(figsize=(16, 5))
counter = Counter(corpus)
most = counter.most_common()
x = []
y = []
for word, count in most[:40]:
    if word not in stop:
        x.append(word)
        y.append(count)

sns.barplot(x=y, y=x)
plt.title("Common words found in all tweets")
plt.savefig('./output/common_tweets.png', bbox_inches='tight')
plt.show()


# Bigram analysis over the tweets.
def get_top_tweet_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda s: s[1], reverse=True)
    return words_freq[:n]


plt.figure(figsize=(10, 5))
top_tweet_bigrams = get_top_tweet_bigrams(train_df['text'])[:10]
x, y = map(list, zip(*top_tweet_bigrams))
sns.barplot(x=y, y=x)
plt.title("Most common bigrams in tweets")
plt.savefig('./output/top_bigrams_tweets.png', bbox_inches='tight')
plt.show()


## Hashtag analysis ##
def show_word_distrib(target=1, field="text"):
    txt = train_df[train_df['target'] == target][field].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(txt)
    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stop)

    rslt = pd.DataFrame(words_except_stop_dist.most_common(20),
                        columns=['Word', 'Frequency']).set_index('Word')
    print(rslt)
    rslt.plot.bar(rot=0)


def find_hashtags(tweet):
    return ", ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or None


def add_hashtags(df):
    df['hashtag'] = df["text"].apply(lambda x: find_hashtags(x))
    df['hashtag'].fillna(value="no", inplace=True)

    return df


df = add_hashtags(train_df)
_l = len([v for v in df.hashtag.values if isinstance(v, str)])
print("-Number of tweets with hashtags: {}".format(_l))
print("\nHashtag distribution in disaster samples: ")
show_word_distrib(target=1, field="hashtag")
plt.title("Frequent hashtags among disaster tweets")
plt.savefig('./output/top_hashtags_disasters.png', bbox_inches='tight')
plt.show()

print("\nHashtag distribution in irrelevant samples: ")
show_word_distrib(target=0, field="hashtag")
plt.title("Frequent hashtags among irrelevant tweets")
plt.savefig('./output/top_hashtags_irrelevant.png', bbox_inches='tight')
plt.show()

## Keyword and Location analysis ##

# Remove the encoded space character for keywords, since appears a lot of times and is junk
df['keyword'] = df['keyword'].map(lambda s: s.replace('%20', ' ') if isinstance(s, str) else s)

un_KW = {kw for kw in df['keyword'].values if isinstance(kw, str)}
tot_KW = len(df) - len(df[df["keyword"].isna()])

un_LOC = {lc for lc in df['location'].values if isinstance(lc, str)}
tot_LOC = len(df) - len(df[df["location"].isna()])

print("\nUnique Keywords: {} out of {}".format(len(un_KW), tot_KW))
print("Samples with no Keyword: {}".format(len(df[df['keyword'].isna()])))

print("Unique Location: {} out of {}".format(len(un_LOC), tot_LOC))
print("Samples with no Location: {}".format(len(df[df['location'].isna()])))

disaster_keywords = [kw for kw in df.loc[df.target == 1].keyword]
regular_keywords = [kw for kw in df.loc[df.target == 0].keyword]

disaster_keywords_counts = dict(pd.DataFrame(data={'x': disaster_keywords}).x.value_counts())
regular_keywords_counts = dict(pd.DataFrame(data={'x': regular_keywords}).x.value_counts())

all_keywords_counts = dict(pd.DataFrame(data={'x': df.keyword.values}).x.value_counts())

print()
# we sort the keywords so the most frequents are on top and we print them with relative
# occurrences in both classes of tweets:
for keyword, _ in sorted(all_keywords_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print("> KW: {}".format(keyword))
    print("-- # in negative tweets: {}".format(disaster_keywords_counts.get(keyword, 0)))
    print("-- # in positive tweets: {}".format(regular_keywords_counts.get(keyword, 0)))
    print('--------')

# Correct the following tweets as they seem to have the wrong target value
ids_with_target_error = [328, 443, 513, 2619, 3640, 3900, 4342, 5781, 6552, 6554, 6570, 6701, 6702, 6729, 6861, 7226]
train_df.at[train_df['id'].isin(ids_with_target_error), 'target'] = 0

### DATA PRE-PROCESSING ###
# Applying the data pre-processing functionality on our train dataset.
print("Cleaning Tweets...")
t0 = time()
train_df["text"] = train_df["text"].apply(lambda s: tweet_cleaner(s))
preprocessing_phase_time = time() - t0
print("End of data preprocessing phase. Time taken: {0:.2f}s".format(preprocessing_phase_time))

# Word Cloud representation of frequent words in disaster tweets after data preprocessing
corpus_new1 = create_corpus(1)
plt.figure(figsize=(12, 8))
word_cloud = WordCloud(
    background_color='black',
    max_font_size=80
).generate(" ".join(corpus_new1))
plt.imshow(word_cloud)
plt.axis('off')
plt.title("Frequent words in disaster tweets")
plt.savefig('./output/word_cloud_disaster.png', bbox_inches='tight')
plt.show()

# Word Cloud representation of frequent words in non-disaster tweets after data preprocessing
corpus_new0 = create_corpus(0)
plt.figure(figsize=(12, 8))
word_cloud = WordCloud(
    background_color='black',
    max_font_size=80
).generate(" ".join(corpus_new0))
plt.imshow(word_cloud)
plt.axis('off')
plt.title("Frequent words in non-disaster tweets")
plt.savefig('./output/word_cloud_irrelevant.png', bbox_inches='tight')
plt.show()

# Word Distribution in disaster tweets
print("\nWord distribution in disaster tweets:")
show_word_distrib(target=1, field="text")
plt.title("Word distribution in disaster tweets:")
plt.savefig('./output/word_distribution_disaster.png', bbox_inches='tight')
plt.show()

# Word Distribution in non-disaster tweets
print("\nWord distribution in irrelevant tweets:")
show_word_distrib(target=0, field="text")
plt.title("Word distribution in irrelevant tweets:")
plt.savefig('./output/word_distribution_irrelevant.png', bbox_inches='tight')
plt.show()

### TRAIN AND TEST SPLIT ###
X = train_df["text"]
Y = train_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

# COUNT VECTORIZER
X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# TF/IDF
X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


### VISUALIZE THE WORD EMBEDINGS ###
def plot_LSA(test_data, test_labels, plot=True):
    """Latent semantic analysis for analyzing relationships between test data
       and the target"""
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue']
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        orange_patch = mpatches.Patch(color='orange', label='Not')
        blue_patch = mpatches.Patch(color='blue', label='Real')
        plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})


fig = plt.figure(figsize=(16, 16))
plot_LSA(X_train_counts, y_train)
plt.title("Visualize data with Count Vectorizer")
plt.savefig('./output/lsa_countvec.png', bbox_inches='tight')
plt.show()

fig2 = plt.figure(figsize=(16, 16))
plot_LSA(X_train_tfidf, y_train)
plt.title("Visualize data with TF-IDF")
plt.savefig('./output/lsa_tfidf.png', bbox_inches='tight')
plt.show()

## COUNT VECTORIZER SCORES

# declare models
clf = LogisticRegression()
clf1 = RidgeClassifier()
clf2 = SGDClassifier()
clf3 = LinearSVC()
clf4 = MultinomialNB()
clf5 = ComplementNB()
clf6 = BernoulliNB()
clf7 = SVC()
clf8 = PassiveAggressiveClassifier()
clf9 = MLPClassifier()
clf10 = RandomForestClassifier()

# declare one array for each model in which we will score the cross validation scores for each individual cv
cv_scores = []
cv_scores1 = []
cv_scores2 = []
cv_scores3 = []
cv_scores4 = []
cv_scores5 = []
cv_scores6 = []
cv_scores7 = []
cv_scores8 = []
cv_scores9 = []
cv_scores10 = []

# try cv from 3 to 10 for each of the models, the average of each score array is then calculated for that cv
# and appended to the cv_scores list along with its correspondent index so we can then later
# find which cv worked best for each different model. We do this process for tweaking the parameters of our models
for x in range(3, 4):
    kfold = KFold(n_splits=x)

    scores = cross_val_score(clf, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores.append([kfold, scores])

    scores1 = cross_val_score(clf1, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores1.append([kfold, scores1])

    scores2 = cross_val_score(clf2, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores2.append([kfold, scores2])

    scores3 = cross_val_score(clf3, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores3.append([kfold, scores3])

    scores4 = cross_val_score(clf4, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores4.append([kfold, scores4])

    scores5 = cross_val_score(clf5, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores5.append([kfold, scores5])

    scores6 = cross_val_score(clf6, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores6.append([kfold, scores6])

    scores7 = cross_val_score(clf7, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores7.append([kfold, scores7])

    scores8 = cross_val_score(clf8, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores8.append([kfold, scores8])

    scores9 = cross_val_score(clf9, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores9.append([kfold, scores9])

    scores10 = cross_val_score(clf10, X_train_counts, y_train, cv=kfold, scoring="f1").mean()
    cv_scores10.append([kfold, scores10])

# We simply reverse sort our lists according to the f1 average score
ordered_cv_scores = (sorted(cv_scores, key=lambda i: i[1], reverse=True))
ordered_cv_scores1 = (sorted(cv_scores1, key=lambda i: i[1], reverse=True))
ordered_cv_scores2 = (sorted(cv_scores2, key=lambda i: i[1], reverse=True))
ordered_cv_scores3 = (sorted(cv_scores3, key=lambda i: i[1], reverse=True))
ordered_cv_scores4 = (sorted(cv_scores4, key=lambda i: i[1], reverse=True))
ordered_cv_scores5 = (sorted(cv_scores5, key=lambda i: i[1], reverse=True))
ordered_cv_scores6 = (sorted(cv_scores6, key=lambda i: i[1], reverse=True))
ordered_cv_scores7 = (sorted(cv_scores7, key=lambda i: i[1], reverse=True))
ordered_cv_scores8 = (sorted(cv_scores8, key=lambda i: i[1], reverse=True))
ordered_cv_scores9 = (sorted(cv_scores9, key=lambda i: i[1], reverse=True))
ordered_cv_scores10 = (sorted(cv_scores10, key=lambda i: i[1], reverse=True))

# print the best average score for each model using count vectorizer.
print("\nRESULTS WITH COUNT VECTORIZER")
print("-----------------------------")

print('Average accuracy for LogisticRegression')
print(ordered_cv_scores[0])

print('Average accuracy for RidgeClassifier')
print(ordered_cv_scores1[0])

print('Average accuracy for SGDClassifier')
print(ordered_cv_scores2[0])

print('Average accuracy for Linear SVC')
print(ordered_cv_scores3[0])

print('Average accuracy for MultinomialNB')
print(ordered_cv_scores4[0])

print('Average accuracy for ComplementNB')
print(ordered_cv_scores5[0])

print('Average accuracy for BernoulliNB')
print(ordered_cv_scores6[0])

print('Average accuracy for SVC')
print(ordered_cv_scores7[0])

print('Average accuracy for PassiveAggressiveClassifier')
print(ordered_cv_scores8[0])

print('Average accuracy for MLPClassifier')
print(ordered_cv_scores9[0])

print('Average accuracy for RandomForestClassifier')
print(ordered_cv_scores10[0])

# Now we prepare our models for prediction
clf.fit(X_train_counts, y_train)
clf1.fit(X_train_counts, y_train)
clf2.fit(X_train_counts, y_train)
clf3.fit(X_train_counts, y_train)
clf4.fit(X_train_counts, y_train)
clf5.fit(X_train_counts, y_train)
clf6.fit(X_train_counts, y_train)
clf7.fit(X_train_counts, y_train)
clf8.fit(X_train_counts, y_train)
clf9.fit(X_train_counts, y_train)
clf10.fit(X_train_counts, y_train)

y_predicted_counts = clf.predict(X_test_counts)
y_predicted_counts1 = clf1.predict(X_test_counts)
y_predicted_counts2 = clf2.predict(X_test_counts)
y_predicted_counts3 = clf3.predict(X_test_counts)
y_predicted_counts4 = clf4.predict(X_test_counts)
y_predicted_counts5 = clf5.predict(X_test_counts)
y_predicted_counts6 = clf6.predict(X_test_counts)
y_predicted_counts7 = clf7.predict(X_test_counts)
y_predicted_counts8 = clf8.predict(X_test_counts)
y_predicted_counts9 = clf9.predict(X_test_counts)
y_predicted_counts10 = clf10.predict(X_test_counts)

print("\nPREDICTION SCORES WITH COUNT VECTORIZER")
print("---------------------------------------")

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("LogisticRegression: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
    accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts1)
print(
    "RidgeClassifier: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts2)
print("SGDClassifier: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts3)
print("Linear SVC: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts4)
print("MultinomialNB: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts5)
print("ComplementNB: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts6)
print("BernoulliNB: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts7)
print("SVC: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts8)
print("PassiveAggressiveClassifier: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
    accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts9)
print("MLPClassifier: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts10)
print("RandomForestClassifier: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
    accuracy, precision, recall, f1))

# TF-IDF SCORES
clf_tfidf = LogisticRegression()
clf_tfidf1 = RidgeClassifier()
clf_tfidf2 = SGDClassifier()
clf_tfidf3 = LinearSVC()
clf_tfidf4 = MultinomialNB()
clf_tfidf5 = ComplementNB()
clf_tfidf6 = BernoulliNB()
clf_tfidf7 = SVC()
clf_tfidf8 = PassiveAggressiveClassifier()
clf_tfidf9 = MLPClassifier()
clf_tfidf10 = RandomForestClassifier()

cv_scores = []
cv_scores1 = []
cv_scores2 = []
cv_scores3 = []
cv_scores4 = []
cv_scores5 = []
cv_scores6 = []
cv_scores7 = []
cv_scores8 = []
cv_scores9 = []
cv_scores10 = []

for x in range(3, 4):  # THIS TAKES VERY LONG, CONSIDER LOWERING THE NUMBER OF ITERATIONS WHEN RUNNING
    kfold = KFold(n_splits=x)

    scores = cross_val_score(clf_tfidf, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores.append([kfold, scores])

    scores1 = cross_val_score(clf_tfidf1, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores1.append([kfold, scores1])

    scores2 = cross_val_score(clf_tfidf2, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores2.append([kfold, scores2])

    scores3 = cross_val_score(clf_tfidf3, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores3.append([kfold, scores3])

    scores4 = cross_val_score(clf_tfidf4, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores4.append([kfold, scores4])

    scores5 = cross_val_score(clf_tfidf5, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores5.append([kfold, scores5])

    scores6 = cross_val_score(clf_tfidf6, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores6.append([kfold, scores6])

    scores7 = cross_val_score(clf_tfidf7, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores7.append([kfold, scores7])

    scores8 = cross_val_score(clf_tfidf8, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores8.append([kfold, scores8])

    scores9 = cross_val_score(clf_tfidf9, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores9.append([kfold, scores9])

    scores10 = cross_val_score(clf_tfidf10, X_train_tfidf, y_train, cv=kfold, scoring="f1").mean()
    cv_scores10.append([kfold, scores10])

# We simply reverse sort our lists according to the f1 average score
ordered_cv_scores = (sorted(cv_scores, key=lambda i: i[1], reverse=True))
ordered_cv_scores1 = (sorted(cv_scores1, key=lambda i: i[1], reverse=True))
ordered_cv_scores2 = (sorted(cv_scores2, key=lambda i: i[1], reverse=True))
ordered_cv_scores3 = (sorted(cv_scores3, key=lambda i: i[1], reverse=True))
ordered_cv_scores4 = (sorted(cv_scores4, key=lambda i: i[1], reverse=True))
ordered_cv_scores5 = (sorted(cv_scores5, key=lambda i: i[1], reverse=True))
ordered_cv_scores6 = (sorted(cv_scores6, key=lambda i: i[1], reverse=True))
ordered_cv_scores7 = (sorted(cv_scores7, key=lambda i: i[1], reverse=True))
ordered_cv_scores8 = (sorted(cv_scores8, key=lambda i: i[1], reverse=True))
ordered_cv_scores9 = (sorted(cv_scores9, key=lambda i: i[1], reverse=True))
ordered_cv_scores10 = (sorted(cv_scores10, key=lambda i: i[1], reverse=True))

# print the best average score for each model using count vectorizer.
print("\nRESULTS WITH TF-IDF")
print("-------------------")

print('Average accuracy for LogisticRegression')
print(ordered_cv_scores[0])

print('Average accuracy for RidgeClassifier')
print(ordered_cv_scores1[0])

print('Average accuracy for SGDClassifier')
print(ordered_cv_scores2[0])

print('Average accuracy for Linear SVC')
print(ordered_cv_scores3[0])

print('Average accuracy for MultinomialNB')
print(ordered_cv_scores4[0])

print('Average accuracy for ComplementNB')
print(ordered_cv_scores5[0])

print('Average accuracy for BernoulliNB')
print(ordered_cv_scores6[0])

print('Average accuracy for SVC')
print(ordered_cv_scores7[0])

print('Average accuracy for PassiveAggressiveClassifier')
print(ordered_cv_scores8[0])

print('Average accuracy for MLPClassifier')
print(ordered_cv_scores9[0])

print('Average accuracy for RandomForestClassifier')
print(ordered_cv_scores10[0])

# Now we prepare our models for prediction
clf_tfidf.fit(X_train_tfidf, y_train)
clf_tfidf1.fit(X_train_tfidf, y_train)
clf_tfidf2.fit(X_train_tfidf, y_train)
clf_tfidf3.fit(X_train_tfidf, y_train)
clf_tfidf4.fit(X_train_tfidf, y_train)
clf_tfidf5.fit(X_train_tfidf, y_train)
clf_tfidf6.fit(X_train_tfidf, y_train)
clf_tfidf7.fit(X_train_tfidf, y_train)
clf_tfidf8.fit(X_train_tfidf, y_train)
clf_tfidf9.fit(X_train_tfidf, y_train)
clf_tfidf10.fit(X_train_tfidf, y_train)

y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
y_predicted_tfidf1 = clf_tfidf1.predict(X_test_tfidf)
y_predicted_tfidf2 = clf_tfidf2.predict(X_test_tfidf)
y_predicted_tfidf3 = clf_tfidf3.predict(X_test_tfidf)
y_predicted_tfidf4 = clf_tfidf4.predict(X_test_tfidf)
y_predicted_tfidf5 = clf_tfidf5.predict(X_test_tfidf)
y_predicted_tfidf6 = clf_tfidf6.predict(X_test_tfidf)
y_predicted_tfidf7 = clf_tfidf7.predict(X_test_tfidf)
y_predicted_tfidf8 = clf_tfidf8.predict(X_test_tfidf)
y_predicted_tfidf9 = clf_tfidf9.predict(X_test_tfidf)
y_predicted_tfidf10 = clf_tfidf10.predict(X_test_tfidf)

print("\nPREDICTION SCORES WITH TF-IDF")
print("-----------------------------")

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf)
print("LogisticRegression: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
    accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf1)
print(
    "RidgeClassifier: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf2)
print("SGDClassifier: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf3)
print("Linear SVC: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf4)
print("MultinomialNB: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf5)
print("ComplementNB: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf6)
print("BernoulliNB: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf7)
print("SVC: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf8)
print("PassiveAggressiveClassifier: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
    accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf9)
print("MLPClassifier: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_tfidf10)
print("RandomForestClassifier: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
    accuracy, precision, recall, f1))


# CONFUSION MATRICES FOR OUR WORD EMBEDDINGS
def plot_confusion_matrix(y_true, y_pred, title, figsize=(5, 5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)


# Work only with logistic regression which yiels the best results for both word embeddings
fig = plt.figure(figsize=(10, 10))
plot_confusion_matrix(y_test, y_predicted_counts, 'Confusion matrix for Count Vectorizer', figsize=(7, 7))
plt.savefig('./output/cm_countvec.png', bbox_inches='tight')
plt.show()

fig2 = plt.figure(figsize=(10, 10))
plot_confusion_matrix(y_test, y_predicted_tfidf, 'Confusion matrix for TF-IDF', figsize=(7, 7))
plt.savefig('./output/cm_tfidf.png', bbox_inches='tight')
plt.show()

## GLOVE FOR TWITTER (A DIFFERENT APPROACH)
glove_twitter = api.load("glove-twitter-50")  # download the model and return as object ready for use


def get_w2v_general(tweet, size, vectors):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tweet.split():
        try:
            vec += vectors[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    return vec


x_train_glove, x_test_glove, y_train_glove, y_test_glove = train_test_split(X, Y, test_size=0.15)

train_vecs_glove_sum = scale(np.concatenate([get_w2v_general(z, 50, glove_twitter) for z in x_train_glove]))
test_vecs_glove_sum = scale(np.concatenate([get_w2v_general(z, 50, glove_twitter) for z in x_test_glove]))

clf = LogisticRegression()
clf.fit(train_vecs_glove_sum, y_train_glove)
y_pred_glove = clf.predict(test_vecs_glove_sum)

accuracy_glove, precision_glove, recall_glove, f1_glove = get_metrics(y_test_glove, y_pred_glove)
print("\nGloVe-50d: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_glove, precision_glove,
                                                                                    recall_glove, f1_glove))

fig_cm_glove = plt.figure(figsize=(10, 10))
plot_confusion_matrix(y_test_glove, y_pred_glove, 'Confusion matrix for GloVe model', figsize=(7, 7))
plt.savefig('./output/cm_glove.png', bbox_inches='tight')
plt.show()

fig_lsa_glove = plt.figure(figsize=(16, 16))
plot_LSA(train_vecs_glove_sum, y_train_glove)
plt.savefig('./output/lsa_glove.png', bbox_inches='tight')
plt.show()

sys.stdout.close()
