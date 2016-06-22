---
layout: post
title: "How to read: Character level deep learning"
date: 2016-06-22
excerpt: "How to make character level deep nets for NLP tasks"
tags: [dnn, nlp, char-level, rnn]
comments: true
---



How to read: Character level deep learning
===================================
2016, the year of the chat bots. Chat bots seem to be extremely popular these days, every other tech company is announcing some form of intelligent language interface. The truth is that language is everywhere, it’s the way we communicate and the way we manage our thoughts. Most, if not all, of our culture & knowledge is encoded and stored in some language. One can think that if you manage to tap to that source of information efficiently then we are definitely a step closer to create ground breaking machine learning algorithms. Of course, chat-bots are not even close to “solving” the language problem, after all language is as broad as our thoughts. But, on the other hand researchers still make useful NLP application that are super cool, like gmail [auto-reply](http://arxiv.org/abs/1606.04870) or [deep-text](https://code.facebook.com/posts/181565595577955) from Facebook.

So after reading a few papers about NLP, and specifically deep learning applications, I decided to go ahead and try out a few things on my own. In this post will demonstrate a few fun character level models for sentiment classification. The models are built with my favourite framework [*Keras*](http://keras.io) (with [Tensorflow](https://tensorflow.org) as back-end). In case you haven’t used *Keras* before I strongly suggest it, it is simple and allows for very fast prototyping (thanks [François Chollet](https://twitter.com/fchollet). After version 1.0 with the new functional API creating complex models can be as easy as a few lines. I’m hoping to demonstrate some of it’s potential as we go along.

If you want to get familiar with the framework I would suggest following the links:

 - [30 seconds to *Keras*](http://keras.io/#getting-started-30-seconds-to-keras)
 - [*Keras* examples](https://github.com/fchollet/keras/tree/master/examples)
 - [Building powerful image classification models using very little data](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

I assume that most people reading this have some basic knowledge about convolution networks, mlps, and rnn/lstm models. 

A very popular post from [Andrej Karpathy](https://twitter.com/karpathy) talking about the effectiveness of recurrent nets presents a character level language model build with RNNs, find it [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). A simple implementation of this model for *Keras*, [here](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py).

Karpathy’s blog post was probably my first encounter with character level modelling and I have to say I was fascinated by it’s performance. In English character level might not be as appealing as other languages like Greek(my native language). Trying to build vocabularies in Greek can be a bit tricky since words change given the context, so working on character level and letting your model figure out the different word permutations is a very appealing property.

If you have a question about a model, the best thing to do with it is experiment. So let's do it!
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>

## What is a character level model?
Let's assume the typical problem of sentiment analysis, given a text, for a example a movie review we need to figure out if the review is positive(1) or negative(0). Let's denote $x_i$ the text input, which is a sequence of words, and $y_i$ the corresponding sentiment, so we create a network $f(x_i)$ that will predict the label of the sample. In such settings a typical approach is to split the text into a sequence of words, and then learn some fixed length embedding of the sequence that will be used to classify it.

![Simple RNN scheme for sentiment](https://raw.githubusercontent.com/offbit/offbit.github.io/master/assets/char-models/lstm.jpg "Simple RNN  scheme for sentiment classification")

In a recurrent model like the above, each word is encoded as a vector (a very nice explanation of word embeddings can be found in Christopher Olah [blog post](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) - along with an explanatory post for [LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - *highly recommended reads*)

Like any typical model that uses a word as it's smallest input entity, a character level model will use the character as the the smallest entity. E.g:
![Character level RNN](https://raw.githubusercontent.com/offbit/offbit.github.io/master/assets/char-models/char-lstm.jpg "character level rnn")

This model is reading characters one by one, to create an embedding of the of a given sentence/text. As such our neural network will try to learn that specific sequences of letters form words separated by spaces or other punctuation points. A paper from A. Karpathy & J. Johnson, ["Visualizing and Understanding Recurrent Networks"](http://arxiv.org/abs/1506.02078), demonstrates visually some of the internal processes of char-rnn models. 

In a paper the ["Exploring the Limits of Language Modeling"](https://arxiv.org/pdf/1602.02410.pdf), from the Google Brain team they show that a character level language model can significantly outperform state of the art models. In their paper the best performing model combines an LSTM with CNN input over the characters, the figure bellow is taken from their paper:
![cnn lstm](https://raw.githubusercontent.com/offbit/offbit.github.io/master/assets/char-models/char-cnn-lstm-google.png "cnn lstm")

In his paper ["Text Understanding from Scratch"](https://arxiv.org/pdf/1502.01710v5.pdf) Zhang et. al. uses pure character level convolution networks to perform text classification with impressive performance. The following figure from his paper describes the model:
![Character level cnn model](https://lh3.googleusercontent.com/-I_Nu_jMK9Cw/V2Q2ddX2zvI/AAAAAAAAG88/GQ0E4vZ4BM4tGmKfTjVLPViAXQvOb0rUQCLcB/s0/Selection_001.png "Char-cnn")

## Building a sentiment model

Let's try build our model on the popular IMDB review database, the labelled data can be found on this Kaggle competition [webpage](https://www.kaggle.com/c/word2vec-nlp-tutorial/data), we are just going to use the labelled labeledTrainData.tsv which contains 25000 reviews with labels. If you haven't worked text before, the competition website offers a nice 4-part tutorial to create sentiment analysis models. 

The base of our model is that we want to encode text from character level, so we'll begin by splitting the text into sentences. Creating sentences from reviews bounds the maximum length of a sequence so it can be easier for our model to handle. After encoding each sentence from characters to a fixed length encoding we use a bi-directional LSTM to read sentence by sentence and create a more elaborate doc encoding. 

The following figure demonstrates the full model

![Full model](https://raw.githubusercontent.com/offbit/offbit.github.io/master/assets/char-models/fullmodel.jpg)

This model starts from reading characters and forming concepts of "words", then uses a bi-directional LSTM to read "words" as a sequence and account for their position. After that each sentence encoding is being passed through a second bi-directional LSTM that does the final document encoding. 

### Preprocessing 

There is minimum preprocessing required for this approach, since our goal is to provide simple text and let the model figure out what that means. So we follow 3 basic steps:

 1. Read review & remove html tags
 2. Clean non English characters 
 3. Split into sentences

{% highlight python %}
data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
txt = ''
docs = []
sentences = []
sentiments = []

for cont, sentiment in zip(data.review, data.sentiment):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean(striphtml(cont)))
    sentences = [sent.lower() for sent in sentences]
    docs.append(sentences)
    sentiments.append(sentiment)
    
{% endhighlight %}
