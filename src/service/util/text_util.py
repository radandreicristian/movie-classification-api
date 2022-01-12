"""A file that contains methods for text pre-processing."""

import json
import pathlib
import string
from typing import Any

import regex as re
from nltk.corpus import stopwords

config_path = pathlib.Path(__file__).parent.parent.absolute() / 'service_config.json'
config = json.load(open(config_path))


def remove_url(text: str) -> Any:
    """
    Remove URL strings from a text.

    :param text: A text string that could contain URLs.
    :return: The text string without URLs.
    """
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_html(text: str) -> Any:
    """
    Remove HTML tags from a string.

    :param text: A text string that could contain HTML tags.
    :return: The text string without HTML tags.
    """
    return re.sub(r'<.*?>', '', text)


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from a string.

    There are multiple ways to remove punctuation, but this method ensures time efficiency.
    :param text: A text string that could contain punctuation.
    :return: The text string without punctuation.
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stop_words(text: str) -> str:
    """
    Remove the stop words from a string.

    :param text: A text string that may contain stop words, which we do not want for our ML model to process.
    :return: The text string without stop words.
    """
    language_stopwords = set(stopwords.words(config['language']))
    text_tokens = text.split()
    return " ".join([x for x in text_tokens if x not in language_stopwords])


def trim(text: str) -> str:
    """
    Remove extra words, keeping the first MAX_LENGTH-1 words in the text.

    This is important since most pre-trained transformers have a token limit (i.e. 512 for BERT-based architectures).
    :param text: A text string with no word limit.
    :return: A text string with at most MAX_LENGTH-1 words.
    """
    words = text.split()
    return " ".join(words[:config['max_len'] - 1])


def remove_whitespaces(text: str) -> Any:
    """
    Remove extra spaces from a string.

    :param text: A text string that may have extra spaces, resulted from from previous preprocessing steps.
    :return: The text string without extra spaces.
    """
    return re.sub(r'\s+', ' ', text)


def preprocess(text: str) -> str:
    """
    Apply multiple preprocessing steps to an input string.

    The preprocessing steps are contained as method references in an array, and the preprocessing is done by iteratively
    calling the steps, in a chaining fashion.
    """
    steps = [remove_url, remove_html, remove_stop_words, remove_punctuation, remove_whitespaces, trim]
    for step in steps:
        text = step(text)
    return text
