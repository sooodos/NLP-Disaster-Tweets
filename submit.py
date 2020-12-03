"""
    Run this file for submitting a score to  the competition.

    Authors: Marcos Antonios Charalambous (mchara01@cs.ucy.ac.cy)
             Sotiris Loizidis (sloizi02@cs.ucy.ac.cy)

    Date: 26/04/2020
"""
import pandas as pd
from sklearn import feature_extraction, linear_model

import re  # For regular expression operations.
import unicodedata  # Character properties for all unicode characters.
import string  # Tools to manipulate strings.
import html  # For html manipulation (unescape)
import twokenize  # Twitter tokenizer
import spacy  # Advanced operations for natural language processing.
from nltk.stem import WordNetLemmatizer  # Lemmatizer for tweet processing
from nltk.tokenize import word_tokenize
from stop_words import safe_get_stop_words

from bs4 import BeautifulSoup  # For decoding HTML to general text.

from ekphrasis.classes.segmenter import Segmenter  # ekphrasis module segmenter for tweets

from abbreviation import abbreviations  # various abbreviations collected


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
    if lowercase:  # convert all text to lowercase
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


train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")

# Correct the following tweets as they seem to have the wrong target value
#ids_with_target_error = [328, 443, 513, 2619, 3640, 3900, 4342, 5781, 6552, 6554, 6570, 6701, 6702, 6729, 6861, 7226]
#train_df.at[train_df['id'].isin(ids_with_target_error), 'target'] = 0

train_df["text"] = train_df["text"].apply(lambda s: tweet_cleaner(s))
test_df["text"] = test_df["text"].apply(lambda s: tweet_cleaner(s))


tfidf_vectorizer = feature_extraction.text.TfidfVectorizer()

train_vectors = tfidf_vectorizer.fit_transform(train_df["text"])

test_vectors = tfidf_vectorizer.transform(test_df["text"])

clf = linear_model.LogisticRegression()

clf.fit(train_vectors, train_df["target"])

sample_submission = pd.read_csv("./sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

sample_submission.to_csv("submission.csv", index=False, header=True)
