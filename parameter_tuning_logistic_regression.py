"""
    Description: In this file we try to optimize the model that yields the
    best f1-score for our given dataset which is the Logistic Regression model.
    We do this by defining some parameter ranges for our model and then perform
    an exhaustive search to determine the optimal configuration.

    Authors: Marcos Antonios Charalambous (mchara01@cs.ucy.ac.cy)
             Sotiris Loizidis (sloizi02@cs.ucy.ac.cy)

    Date: 26/04/2020
"""
import sys  # For system calls such us std.out
import pandas as pd  # CSV file I/O (e.g. pd.read_csv).
import warnings  # Omit any warnings in output.

import numpy as np  # Linear algebra.
import re  # For regular expression operations.
import unicodedata  # Character properties for all unicode characters.
import string  # Tools to manipulate strings.
import html
import nltk
import twokenize
import spacy  # Advanced operations for natural language processing.

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from stop_words import safe_get_stop_words
from bs4 import BeautifulSoup  # For decoding HTML to general text.

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from ekphrasis.classes.segmenter import Segmenter  # ekphrasis library segmenter for tweets

from abbreviation import abbreviations  # various abbreviations collected

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

sys.stdout = open("./output/param_tuning.txt", "w")

warnings.filterwarnings("ignore")

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

# Applying the data pre-processing functionality on our train dataset.
# Also creating new columns that will be used for trying to find a better dataset.
train_df["keyword"] = train_df["keyword"].fillna('no_keyword ')
train_df["location"] = train_df["location"].fillna('no_location ')
train_df["loc+text"] = train_df["location"] + train_df["text"]
train_df["key+text"] = train_df["keyword"] + train_df["text"]
train_df["key+loc+text"] = train_df["keyword"] + train_df["location"] + train_df["text"]
train_df["loc+text"] = train_df["loc+text"].apply(lambda s: tweet_cleaner(s))
train_df["key+text"] = train_df["key+text"].apply(lambda s: tweet_cleaner(s))
train_df["key+loc+text"] = train_df["key+loc+text"].apply(lambda s: tweet_cleaner(s))
train_df["text"] = train_df["text"].apply(lambda s: tweet_cleaner(s))

# We will generate scores for 4 different versions of data
# We train the text part only, the text part with the location appended,
# the text with keyword appended and all three appended together.
# As expected since we are working with countVectorizer and TF-IDF
# we managed no improvement, it was just interesting to check.

X_RANGE = ["text", "loc+text", "key+text", "key+loc+text"]
Y = train_df["target"]

for x in X_RANGE:
    print('Results with the following columns:', x)
    print()
    X = train_df[x]

    # Split our train dataset to a 4-to-1 ratio in order to use that as our test set.
    # This is ofcourse done in order to get some immediate results for our report
    # and is not the if we want to register a submission. If we want to do that
    # we must use the full training set for evaluation and the test.csv for testing
    # as determined by the competition rules.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

    ## START OF PARAMETER TUNING

    # The pipeline is an easy way for us to apply quickly and neatly the training of our set
    # After the fit(X,y) function is used the sets are passed through the pipeline and are
    # treated as dictated by the corresponding tuned parameters.

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression())])

    # Here we provide some ranges and different key-values for the
    # parameters used in GridSearch.

    # vect__ngram_range: We want to vectorize with mono-grams,mono-grams
    # and bi-grams or bi-grams alone,.

    # tfidf__use_idf: We want to be able to choose wheter or not to stay
    # with countVectorizer or transform it to tfidf

    # tfidf__norm: Try different norms.

    # clf__C: Try different levels of regularization strength.
    tuned_parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__C': [0.1, 1, 10, 30, 50, 80]
    }

    score = 'f1'
    print("# Tuning hyper-parameters for %s" % score)
    print()
    np.errstate(divide='ignore')

    # We make use of GrisSearchCV which perform an exhaustive search over our
    # different options to try and find the best possible parameters for our
    # model
    clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for mean, std, params in zip(clf.cv_results_['mean_test_score'],
                                 clf.cv_results_['std_test_score'],
                                 clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on a ratio of 85/15 on the train set.")
    print("The scores are computed on the 15% of the train set that was not used for evaluation.")
    print()
    # Calls a prediction on our estimator withe the BEST found parameters.
    print(classification_report(y_test, clf.predict(X_test), digits=4))
    print()
    print('Accuracy:', accuracy_score(y_test, clf.predict(X_test)))
    print()
    print('************************************************************************')

sys.stdout.close()
