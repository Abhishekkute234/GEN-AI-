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

def text*cleaner(text):
rules = [
{r'>\s+': u'>'}, # remove spaces after a tag opens or closes
{r'\s+': u' '}, # replace consecutive spaces
{r'\s*<br\s*/?>\s*': u'\n'}, # newline after a <br>
{r'</(div)\s*>\s*': u'\n'}, # newline after </p> and </div> and <h1/>...
{r'</(p|h\d)\s*>\s*': u'\n\n'}, # newline after </p> and </div> and <h1/>...
{r'<head>._<\s_(/head|body)[^>]_>': u''}, # remove <head> to </head>
{r'<a\s+href="([^"]+)"[^>]_>._</a>': r'\1'}, # show links instead of texts
{r'[ \t]_<[^<]\_?/?>': u''}, # remove remaining tags
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

**Tensorflow implementation**

Tensorflow implementation of the pretrained biLM used to compute ELMo representations from `"Deep contextualized word representations" <http://arxiv.org/abs/1802.05365>`\_\_.

This repository supports both training biLMs and using pre-trained models for prediction.

We also have a pytorch implementation available in `AllenNLP <http://allennlp.org/>`\_\_.

You may also find it easier to use the version provided in `Tensorflow Hub <https://www.tensorflow.org/hub/modules/google/elmo/2>`\_\_ if you just like to make predictions.

**pre-trained models:**

We have got several pre-trained English language biLMs available for use. Each model is specified with two separate files, a JSON formatted "options" file with hyperparameters and a hdf5 formatted file with the model weights. Links to the pre-trained models are available `here <https://allennlp.org/elmo>`\_\_.

There are three ways to integrate ELMo representations into a downstream task, depending on your use case.

1. Compute representations on the fly from raw text using character input. This is the most general method and will handle any input text. It is also the most computationally expensive.
2. Precompute and cache the context independent token representations, then compute context dependent representations using the biLSTMs for input data. This method is less computationally expensive then #1, but is only applicable with a fixed, prescribed vocabulary.
3. Precompute the representations for your entire dataset and save to a file.

We have used all of these methods in the past for various use cases. #1 is necessary for evaluating at test time on unseen data (e.g. public SQuAD leaderboard). #2 is a good compromise for large datasets where the size of the file in is unfeasible (SNLI, SQuAD). #3 is a good choice for smaller datasets or in cases where you'd like to use ELMo in other frameworks.

In all cases, the process roughly follows the same steps. First, create a `Batcher` (or `TokenBatcher` for #2) to translate tokenized strings to numpy arrays of character (or token) ids. Then, load the pretrained ELMo model (class `BidirectionalLanguageModel`). Finally, for steps #1 and #2 use `weight_layers` to compute the final ELMo representations. For #3, use `BidirectionalLanguageModel` to write all the intermediate layers to a file.

.. figure:: docs/pic/ngram_cnn_highway_1.png
Architecture of the language model applied to an example sentence [Reference: `arXiv paper <https://arxiv.org/pdf/1508.06615.pdf>`__].

.. figure:: docs/pic/Glove_VS_DCWE.png

---

## FastText

.. figure:: docs/pic/fasttext-logo-color-web.png

fastText is a library for efficient learning of word representations and sentence classification.

**Github:** `facebookresearch/fastText <https://github.com/facebookresearch/fastText>`\_\_

**Models**

- Recent state-of-the-art `English word vectors <https://fasttext.cc/docs/en/english-vectors.html>`\_\_.
- Word vectors for `157 languages trained on Wikipedia and Crawl <https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md>`\_\_.
- Models for `language identification <https://fasttext.cc/docs/en/language-identification.html#content>`** and `various supervised tasks <https://fasttext.cc/docs/en/supervised-models.html#content>`**.

**Supplementary data :**

- The preprocessed `YFCC100M data <https://fasttext.cc/docs/en/dataset.html#content>`\_\_ .

**FAQ**

You can find `answers to frequently asked questions <https://fasttext.cc/docs/en/faqs.html#content>`** on Their project `website <https://fasttext.cc/>`**.

**Cheatsheet**

Also a `cheatsheet <https://fasttext.cc/docs/en/cheatsheet.html#content>`\_\_ is provided full of useful one-liners.

```
Weighted Words
```

---

## Term frequency

Term frequency is Bag of words that is one of the simplest techniques of text feature extraction. This method is based on counting number of the words in each document and assign it to feature space.

---

## Term Frequency-Inverse Document Frequency

The mathematical representation of weight of a term in a document by Tf-idf is given:

.. image:: docs/eq/tf-idf.gif
:width: 10px

Where N is number of documents and df(t) is the number of documents containing the term t in the corpus. The first part would improve recall and the later would improve the precision of the word embedding. Although tf-idf tries to overcome the problem of common terms in document, it still suffers from some other descriptive limitations. Namely, tf-idf cannot account for the similarity between words in the document since each word is presented as an index. In the recent years, with development of more complex models, such as neural nets, new methods has been presented that can incorporate concepts, such as similarity of words and part of speech tagging. This work uses, word2vec and Glove, two of the most common methods that have been successfully used for deep learning techniques.

.. code:: python

    from sklearn.feature_extraction.text import TfidfVectorizer
    def loadData(X_train, X_test,MAX_NB_WORDS=75000):
        vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
        X_train = vectorizer_x.fit_transform(X_train).toarray()
        X_test = vectorizer_x.transform(X_test).toarray()
        print("tf-idf with",str(np.array(X_train).shape[1]),"features")
        return (X_train,X_test)

```
Comparison of Feature Extraction Techniques
```

+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| **Model** | **Advantages** | **Limitation** |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| **Weighted Words** | _ Easy to compute | _ It does not capture the position in the text (syntactic) |
| | | |
| | _ Easy to compute the similarity between 2 documents using it | _ It does not capture meaning in the text (semantics) |
| | | |
| | _ Basic metric to extract the most descriptive terms in a document | |
| | | _ Common words effect on the results (e.g., “am”, “is”, etc.) |
| | _ Works with an unknown word (e.g., New words in languages) | |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| **TF-IDF** | _ Easy to compute | _ It does not capture the position in the text (syntactic) |
| | | |
| | | |
| | _ Easy to compute the similarity between 2 documents using it | _ It does not capture meaning in the text (semantics) |
| | | |
| | | |
| | _ Basic metric to extract the most descriptive terms in a document | |
| | | |
| | | |
| | _ Common words do not affect the results due to IDF (e.g., “am”, “is”, etc.) | |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| **Word2Vec** | _ It captures the position of the words in the text (syntactic) | _ It cannot capture the meaning of the word from the text (fails to capture polysemy) |
| | | |
| | _ It captures meaning in the words (semantics) | _ It cannot capture out-of-vocabulary words from corpus |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| **GloVe (Pre-Trained)** | _ It captures the position of the words in the text (syntactic) | _ It cannot capture the meaning of the word from the text (fails to capture polysemy) |
| | | |
| | _ It captures meaning in the words (semantics) | |
| | | _ Memory consumption for storage |
| | _ Trained on huge corpus | |
| | | |
| | | _ It cannot capture out-of-vocabulary words from corpus |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| **GloVe (Trained)** | _ It is very straightforward, e.g., to enforce the word vectors to capture sub-linear relationships in the vector space (performs better than Word2vec) | _ Memory consumption for storage |
| | | |
| | _ Lower weight for highly frequent word pairs, such as stop words like “am”, “is”, etc. Will not dominate training progress | _ Needs huge corpus to learn |
| | | |
| | | _ It cannot capture out-of-vocabulary words from the corpus |
| | | |
| | | _ It cannot capture the meaning of the word from the text (fails to capture polysemy) |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| **FastText** | _ Works for rare words (rare in their character n-grams which are still shared with other words | _ It cannot capture the meaning of the word from the text (fails to capture polysemy) |
| | | |
| | | _ Memory consumption for storage |
| | _ Solves out of vocabulary words with n-gram in character level | |
| | | _ Computationally is more expensive in comparing with GloVe and Word2Vec |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
|**Contextualized Word Representations**| _ It captures the meaning of the word from the text (incorporates context, handling polysemy) | _ Memory consumption for storage |
| | | |
| | | _ Improves performance notably on downstream tasks. Computationally is more expensive in comparison to others |
| | | |
| | | _ Needs another word embedding for all LSTM and feedforward layers |
| | | |
| | | _ It cannot capture out-of-vocabulary words from a corpus |
| | | |
| | | |
| | | _ Works only sentence and document level (it cannot work for individual word level) |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
========================
Dimensionality Reduction
========================

---

```
Principal Component Analysis (PCA)
```

Principle component analysis~(PCA) is the most popular technique in multivariate analysis and dimensionality reduction. PCA is a method to identify a subspace in which the data approximately lies. This means finding new variables that are uncorrelated and maximizing the variance to preserve as much variability as possible.

Example of PCA on text dataset (20newsgroups) from tf-idf with 75000 features to 2000 components:

.. code:: python

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
        vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
        X_train = vectorizer_x.fit_transform(X_train).toarray()
        X_test = vectorizer_x.transform(X_test).toarray()
        print("tf-idf with", str(np.array(X_train).shape[1]), "features")
        return (X_train, X_test)


    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    X_train,X_test = TFIDF(X_train,X_test)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2000)
    X_train_new = pca.fit_transform(X_train)
    X_test_new = pca.transform(X_test)

    print("train with old features: ",np.array(X_train).shape)
    print("train with new features:" ,np.array(X_train_new).shape)

    print("test with old features: ",np.array(X_test).shape)
    print("test with new features:" ,np.array(X_test_new).shape)

output:

.. code:: python

    tf-idf with 75000 features
    train with old features:  (11314, 75000)
    train with new features: (11314, 2000)
    test with old features:  (7532, 75000)
    test with new features: (7532, 2000)

```
Linear Discriminant Analysis (LDA)
```

Linear Discriminant Analysis (LDA) is another commonly used technique for data classification and dimensionality reduction. LDA is particularly helpful where the within-class frequencies are unequal and their performances have been evaluated on randomly generated test data. Class-dependent and class-independent transformation are two approaches in LDA where the ratio of between-class-variance to within-class-variance and the ratio of the overall-variance to within-class-variance are used respectively.

.. code:: python

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
X_train = vectorizer_x.fit_transform(X_train).toarray()
X_test = vectorizer_x.transform(X_test).toarray()
print("tf-idf with", str(np.array(X_train).shape[1]), "features")
return (X_train, X_test)

from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

X_train,X_test = TFIDF(X_train,X_test)

LDA = LinearDiscriminantAnalysis(n_components=15)
X_train_new = LDA.fit(X_train,y_train)
X_train_new = LDA.transform(X_train)
X_test_new = LDA.transform(X_test)

print("train with old features: ",np.array(X_train).shape)
print("train with new features:" ,np.array(X_train_new).shape)

print("test with old features: ",np.array(X_test).shape)
print("test with new features:" ,np.array(X_test_new).shape)

output:

.. code::

    tf-idf with 75000 features
    train with old features:  (11314, 75000)
    train with new features: (11314, 15)
    test with old features:  (7532, 75000)
    test with new features: (7532, 15)

```
Non-negative Matrix Factorization (NMF)
```

.. code:: python

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from sklearn.decomposition import NMF


    def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
        vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
        X_train = vectorizer_x.fit_transform(X_train).toarray()
        X_test = vectorizer_x.transform(X_test).toarray()
        print("tf-idf with", str(np.array(X_train).shape[1]), "features")
        return (X_train, X_test)


    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    X_train,X_test = TFIDF(X_train,X_test)



    NMF_ = NMF(n_components=2000)
    X_train_new = NMF_.fit(X_train)
    X_train_new =  NMF_.transform(X_train)
    X_test_new = NMF_.transform(X_test)

    print("train with old features: ",np.array(X_train).shape)
    print("train with new features:" ,np.array(X_train_new).shape)

    print("test with old features: ",np.array(X_test).shape)
    print("test with new features:" ,np.array(X_test_new))

output:

.. code::

    tf-idf with 75000 features
    train with old features:  (11314, 75000)
    train with new features: (11314, 2000)
    test with old features:  (7532, 75000)
    test with new features: (7532, 2000)

```
Random Projection
```

Random projection or random feature is a dimensionality reduction technique mostly used for very large volume dataset or very high dimensional feature space. Text and document, especially with weighted feature extraction, can contain a huge number of underlying features.
Many researchers addressed Random Projection for text data for text mining, text classification and/or dimensionality reduction.
We start to review some random projection techniques.

.. image:: docs/pic/Random%20Projection.png

.. code:: python

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
        vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
        X_train = vectorizer_x.fit_transform(X_train).toarray()
        X_test = vectorizer_x.transform(X_test).toarray()
        print("tf-idf with", str(np.array(X_train).shape[1]), "features")
        return (X_train, X_test)


    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    X_train,X_test = TFIDF(X_train,X_test)

    from sklearn import random_projection

    RandomProjection = random_projection.GaussianRandomProjection(n_components=2000)
    X_train_new = RandomProjection.fit_transform(X_train)
    X_test_new = RandomProjection.transform(X_test)

    print("train with old features: ",np.array(X_train).shape)
    print("train with new features:" ,np.array(X_train_new).shape)

    print("test with old features: ",np.array(X_test).shape)
    print("test with new features:" ,np.array(X_test_new).shape)

output:

.. code:: python

    tf-idf with 75000 features
    train with old features:  (11314, 75000)
    train with new features: (11314, 2000)
    test with old features:  (7532, 75000)
    test with new features: (7532, 2000)

```
Autoencoder
```

Autoencoder is a neural network technique that is trained to attempt to map its input to its output. The autoencoder as dimensional reduction methods have achieved great success via the powerful reprehensibility of neural networks. The main idea is, one hidden layer between the input and output layers with fewer neurons can be used to reduce the dimension of feature space. Specially for texts, documents, and sequences that contains many features, autoencoder could help to process data faster and more efficiently.

.. image:: docs/pic/Autoencoder.png

.. code:: python

from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations

encoding_dim = 1500

# this is our input placeholder

input = Input(shape=(n,))

# "encoded" is the encoded representation of the input

encoded = Dense(encoding_dim, activation='relu')(input)

# "decoded" is the lossy reconstruction of the input

decoded = Dense(n, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction

autoencoder = Model(input, decoded)

# this model maps an input to its encoded representation

encoder = Model(input, encoded)

encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model

decoder_layer = autoencoder.layers[-1]

# create the decoder model

decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

Load data:

.. code:: python

autoencoder.fit(x_train, x_train,
epochs=50,
batch_size=256,
shuffle=True,
validation_data=(x_test, x_test))

```
T-distributed Stochastic Neighbor Embedding (T-SNE)
```

T-distributed Stochastic Neighbor Embedding (T-SNE) is a nonlinear dimensionality reduction technique for embedding high-dimensional data which is mostly used for visualization in a low-dimensional space. This approach is based on `G. Hinton and ST. Roweis <https://www.cs.toronto.edu/~fritz/absps/sne.pdf>`\_\_ . SNE works by converting the high dimensional Euclidean distances into conditional probabilities which represent similarities.

`Example <http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>`\_\_:

.. code:: python

import numpy as np
from sklearn.manifold import TSNE
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded.shape

Example of Glove and T-SNE for text:

.. image:: docs/pic/TSNE.png

===============================
Text Classification Techniques
===============================

---

```
Rocchio classification
```

The first version of Rocchio algorithm is introduced by rocchio in 1971 to use relevance feedback in querying full-text databases. Since then many researchers have addressed and developed this technique for text and document classification. This method uses TF-IDF weights for each informative word instead of a set of Boolean features. Using a training set of documents, Rocchio's algorithm builds a prototype vector for each class which is an average vector over all training document vectors that belongs to a certain class. Then, it will assign each test document to a class with maximum similarity that between test document and each of the prototype vectors.

When in nearest centroid classifier, we used for text as input data for classification with tf-idf vectors, this classifier is known as the Rocchio classifier.

.. code:: python

    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn.pipeline import Pipeline
    from sklearn import metrics
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', NearestCentroid()),
                         ])

    text_clf.fit(X_train, y_train)


    predicted = text_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))

Output:

.. code:: python

                  precision    recall  f1-score   support

              0       0.75      0.49      0.60       319
              1       0.44      0.76      0.56       389
              2       0.75      0.68      0.71       394
              3       0.71      0.59      0.65       392
              4       0.81      0.71      0.76       385
              5       0.83      0.66      0.74       395
              6       0.49      0.88      0.63       390
              7       0.86      0.76      0.80       396
              8       0.91      0.86      0.89       398
              9       0.85      0.79      0.82       397
             10       0.95      0.80      0.87       399
             11       0.94      0.66      0.78       396
             12       0.40      0.70      0.51       393
             13       0.84      0.49      0.62       396
             14       0.89      0.72      0.80       394
             15       0.55      0.73      0.63       398
             16       0.68      0.76      0.71       364
             17       0.97      0.70      0.81       376
             18       0.54      0.53      0.53       310
             19       0.58      0.39      0.47       251

    avg / total       0.74      0.69      0.70      7532
