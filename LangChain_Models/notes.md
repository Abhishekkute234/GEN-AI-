################################################
Text Classification Algorithms: A Survey
################################################

|UniversityCube| |DOI| |Best| |medium| |mendeley| |contributions-welcome| |arXiv| |ansicolortags| |contributors| |twitter|

.. figure:: docs/pic/WordArt.png

Referenced paper : `Text Classification Algorithms: A Survey <https://arxiv.org/abs/1904.08067>`\_\_

|BPW|

##################
Table of Contents
##################
.. contents::
:local:
:depth: 4

============
Introduction
============

.. figure:: docs/pic/OverviewTextClassification.png

====================================
Text and Document Feature Extraction
====================================

---

Text feature extraction and pre-processing for classification algorithms are very significant. In this section, we start to talk about text cleaning since most of documents contain a lot of noise. In this part, we discuss two primary methods of text feature extractions- word embedding and weighted word.

```
Text Cleaning and Pre-processing
```

In Natural Language Processing (NLP), most of the text and documents contain many words that are redundant for text classification, such as stopwords, miss-spellings, slangs, and etc. In this section, we briefly explain some techniques and methods for text cleaning and pre-processing text documents. In many algorithms like statistical and probabilistic learning methods, noise and unnecessary features can negatively affect the overall perfomance. So, elimination of these features are extremely important.

---

## Tokenization

Tokenization is the process of breaking down a stream of text into words, phrases, symbols, or any other meaningful elements called tokens. The main goal of this step is to extract individual words in a sentence. Along with text classifcation, in text mining, it is necessay to incorporate a parser in the pipeline which performs the tokenization of the documents; for example:

sentence:

.. code::

After sleeping for four hours, he decided to sleep for another four

In this case, the tokens are as follows:

.. code::

    {'After', 'sleeping', 'for', 'four', 'hours', 'he', 'decided', 'to', 'sleep', 'for', 'another', 'four'}

Here is python code for Tokenization:

.. code:: python

from nltk.tokenize import word_tokenize
text = "After sleeping for four hours, he decided to sleep for another four"
tokens = word_tokenize(text)
print(tokens)

---

## Stop words

Text and document classification over social media, such as Twitter, Facebook, and so on is usually affected by the noisy nature (abbreviations, irregular forms) of the text corpuses.

Here is an exmple from `geeksforgeeks <https://www.geeksforgeeks.org/removing-stop-words-nltk-python/>`\_\_

.. code:: python

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
if w not in stop_words:
filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)

Output:

.. code:: python

['This', 'is', 'a', 'sample', 'sentence', ',', 'showing',
'off', 'the', 'stop', 'words', 'filtration', '.']
['This', 'sample', 'sentence', ',', 'showing', 'stop',
'words', 'filtration', '.']

---

## Capitalization

Sentences can contain a mixture of uppercase and lower case letters. Multiple sentences make up a text document. To reduce the problem space, the most common approach is to reduce everything to lower case. This brings all words in a document in same space, but it often changes the meaning of some words, such as "US" to "us" where first one represents the United States of America and second one is a pronoun. To solve this, slang and abbreviation converters can be applied.

.. code:: python

text = "The United States of America (USA) or America, is a federal republic composed of 50 states"
print(text)
print(text.lower())

Output:

.. code:: python

"The United States of America (USA) or America, is a federal republic composed of 50 states"
"the united states of america (usa) or america, is a federal republic composed of 50 states"

---

## Slangs and Abbreviations

Slangs and abbreviations can cause problems while executing the pre-processing steps. An abbreviation is a shortened form of a word, such as SVM stand for Support Vector Machine. Slang is a version of language that depicts informal conversation or text that has different meaning, such as "lost the plot", it essentially means that 'they've gone mad'. Common method to deal with these words is converting them to formal language.

---

## Noise Removal

Another issue of text cleaning as a pre-processing step is noise removal. Text documents generally contains characters like punctuations or special characters and they are not necessary for text mining or classification purposes. Although punctuation is critical to understand the meaning of the sentence, but it can affect the classification algorithms negatively.

Here is simple code to remove standard noise from text:

.. code:: python

def text_cleaner(text):
rules = [
{r'>\s+': u'>'}, # remove spaces after a tag opens or closes
{r'\s+': u' '}, # replace consecutive spaces
{r'\s*<br\s*/?>\s*': u'\n'}, # newline after a <br>
{r'</(div)\s*>\s*': u'\n'}, # newline after </p> and </div> and <h1/>...
{r'</(p|h\d)\s*>\s*': u'\n\n'}, # newline after </p> and </div> and <h1/>...
{r'<head>.*<\s*(/head|body)[^>]*>': u''}, # remove <head> to </head>
{r'<a\s+href="([^"]+)"[^>]_>._</a>': r'\1'}, # show links instead of texts
{r'[ \t]_<[^<]_?/?>': u''}, # remove remaining tags
{r'^\s+': u''} # remove spaces at the beginning
]
for rule in rules:
for (k, v) in rule.items():
regex = re.compile(k)
text = regex.sub(v, text)
text = text.rstrip()
return text.lower()

---

## Spelling Correction

An optional part of the pre-processing step is correcting the misspelled words. Different techniques, such as hashing-based and context-sensitive spelling correction techniques, or spelling correction using trie and damerau-levenshtein distance bigram have been introduced to tackle this issue.

.. code:: python

from autocorrect import spell

print spell('caaaar')
print spell(u'mussage')
print spell(u'survice')
print spell(u'hte')

Result:

.. code::

    caesar
    message
    service
    the

---

## Stemming

Text Stemming is modifying a word to obtain its variants using different linguistic processeses like affixation (addition of affixes). For example, the stem of the word "studying" is "study", to which -ing.

Here is an example of Stemming from `NLTK <https://pythonprogramming.net/stemming-nltk-tutorial/>`\_\_

.. code:: python

    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize

    ps = PorterStemmer()

    example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

    for w in example_words:
    print(ps.stem(w))

Result:

.. code::

python
python
python
python
pythonli

---

## Lemmatization

Text lemmatization is the process of eliminating redundant prefix or suffix of a word and extract the base word (lemma).

.. code:: python

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))

```
Word Embedding
```

Different word embedding procedures have been proposed to translate these unigrams into consummable input for machine learning algorithms. A very simple way to perform such embedding is term-frequency~(TF) where each word will be mapped to a number corresponding to the number of occurrence of that word in the whole corpora. The other term frequency functions have been also used that represent word-frequency as Boolean or logarithmically scaled number. Here, each document will be converted to a vector of same length containing the frequency of the words in that document. Although such approach may seem very intuitive but it suffers from the fact that particular words that are used very commonly in language literature might dominate this sort of word representations.

.. image:: docs/pic/CBOW.png

---

## Word2Vec

Original from https://code.google.com/p/word2vec/

I’ve copied it to a github project so that I can apply and track community
patches (starting with capability for Mac OS X
compilation).

- **makefile and some source has been modified for Mac OS X
  compilation** See
  https://code.google.com/p/word2vec/issues/detail?id=1#c5
- **memory patch for word2vec has been applied** See
  https://code.google.com/p/word2vec/issues/detail?id=2
- Project file layout altered

There seems to be a segfault in the compute-accuracy utility.

To get started:

::

cd scripts && ./demo-word.sh

Original README text follows:

This tool provides an efficient implementation of the continuous bag-of-words and skip-gram architectures for computing vector representations of words. These representations can be subsequently used in many natural language processing applications and for further research purposes.

this code provides an implementation of the Continuous Bag-of-Words (CBOW) and
the Skip-gram model (SG), as well as several demo scripts.

Given a text corpus, the word2vec tool learns a vector for every word in
the vocabulary using the Continuous Bag-of-Words or the Skip-Gram neural
network architectures. The user should specify the following: -
desired vector dimensionality (size of the context window for
either the Skip-Gram or the Continuous Bag-of-Words model), training
algorithm (hierarchical softmax and / or negative sampling), threshold
for downsampling the frequent words, number of threads to use,
format of the output word vector file (text or binary).

Usually, other hyper-parameters, such as the learning rate do not
need to be tuned for different training sets.

The script demo-word.sh downloads a small (100MB) text corpus from the
web, and trains a small word vector model. After the training is
finished, users can interactively explore the similarity of the
words.

More information about the scripts is provided at
https://code.google.com/p/word2vec/

---

## Global Vectors for Word Representation (GloVe)

.. image:: /docs/pic/Glove.PNG

An implementation of the GloVe model for learning word representations is provided, and describe how to download web-dataset vectors or train your own. See the `project page <http://nlp.stanford.edu/projects/glove/>`** or the `paper <http://nlp.stanford.edu/pubs/glove.pdf>`** for more information on glove vectors.

---

## Contextualized Word Representations

ELMo is a deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). These word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. They can be easily added to existing models and significantly improve the state of the art across a broad range of challenging NLP problems, including question answering, textual entailment and sentiment analysis.

**ELMo representations are:**

- **Contextual:** The representation for each word depends on the entire context in which it is used.
- **Deep:** The word representations combine all layers of a deep pre-trained neural network.
- **Character based:** ELMo representations are purely character based, allowing the network to use morphological clues to form robust representations for out-of-vocabulary tokens unseen in training.
