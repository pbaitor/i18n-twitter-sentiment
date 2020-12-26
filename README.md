# Multi-languange sentiment analysis using neural networks and zero shot cross lingual transfer learning.

Final project for the Building AI course

*December 2020*

## Summary

TL;DR: this project builds an **AI that predicts 11 emotions from text sentences** and you can try it on real tweets going to https://emotions.aitorperez.com

## Background

I wanted to get a practical experience with:
* Programming a multilayer perceptron (MLP)
* Using a publicly available datasetfor training and testing
* Solving a multi-label classification task
* Putting the AI system to use

After doing some research (mainly about Natural Language Processing, due to personal interest) I discovered 2 things that caught my attention:
* LASER, an AI system from facebook that maps sentences to a 1024-dimensional space with the additional benefit that sentences of similar meaning across 93 languages get mapped nearby on the embedding space
* A dataset of tweets for sentiment analysis which includes annotations for intensity (regression and ordinal classification for 4 emotions), polarity (regression and ordinal classification), presence or absence of emotion (multi-labelling of 11 emotions)

I thought it would be interesting to try to put together an AI capable of doing a given task in any language.

## How is it used?
*Describe the process of using the solution. In what kind situations is the solution needed (environment, time, etc.)? Who are the users, what kinds of needs should be taken into account?*





### Development process

#### Dependencies

```
# Laser and trained models
pip install laserembeddings
python -m laserembeddings download-models

# BM-MLL loss function
pip install bpmll

# Punkt tokenizer from nltk
from nltk import download as nltkdownload
nltkdownload('punkt')
```

#### Preparing the datasets

First, we load the data to numpy arrays from the txt files:
```
import numpy as np

X_subset_raw = np.genfromtxt('/content/drive/MyDrive/Machine Learning/Datasets/SemEval-2018 Affect in Tweets E-c/2018-E-c-En-dev.txt', skip_header=1, usecols=(1), dtype=str, delimiter="\t", comments="#@")
Y_subset = np.genfromtxt('/content/drive/MyDrive/Machine Learning/Datasets/SemEval-2018 Affect in Tweets E-c/2018-E-c-En-dev.txt', skip_header=1, usecols=list(range(2,13)), delimiter="\t", comments="#@")
# numpy.gentext assumes "#" as the start of a comment on text, but the tweets dataset have a lot of # symbols
# We change the comment delimiter to something that won't be found on the tweet text

X_subset = np.array([s.replace('#', '') for s in X_subset_raw])
# Finally we remove all the # (because we want "word" to have the same meaning than "#word")
```

From the paper where BP-MLL is proposed as a loss function to adapt ANN to multi-label classification there is a consideration we must observe:
*Every sample needs to have at least one label and no sample may have all labels.*

We can safely assume no tweet will be annotated with every emotion, but there may be tweets without any.
We can inspect each subset for that:
```
count = 0
for element in Y_subset:
  if np.sum(element) == 0:
    count +=1
print(count)
```

On the training subset we find these tweets without emotion:
```
204
```

To remove this instances, we can use the following:
```
Y_subset_clean = np.array([item for item in Y_subset if np.sum(item)>0])
print("Cleaned Y_subset from ", len(Y_subset), " to ", len(Y_subset_clean))

X_subset_clean = np.array([X_train[i] for i, item in enumerate(Y_subset) if np.sum(item)>0])
print("Cleaned X_subset from ", len(X_subset), " to ", len(X_subset_clean))
```

For the training subset this results in:
```
Cleaned Y_train from  6838  to  6634
Cleaned X_train from  6838  to  6634
```

Finally, we run a quick statistic to check the distribution of the annotated emotions on the training subset:
```
accum = sum(Y_train_clean)
print(np.around(accum/len(Y_train_clean),decimals=2))
```

The results are:

| anger | anticipation | disgust | fear | joy | love | optimism | pessimism | sadness | surprise | trust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.38 | 0.15 | 0.39 | 0.19 | 0.37 | 0.11 | 0.30 | 0.12 | 0.30 | 0.05 | 0.05 |

Clearly the training dataset isn't balanced and there are some emotions with very few data available (surprise, trust).

#### Extracting embeddings

The embeddings on Laser's 1024-dimensional space can be easily obtaine with:
```
from laserembeddings import Laser
laser = Laser()
X_train_embeddings = laser.embed_sentences(X_train_clean,lang='en')
```

This results in a matrix with this shape for the trainig dataset:
```
(6634, 1024)
```

#### Training the multi-label classifier

We first define the following MLP using as a template the example one from the BP-MLL implementation.
* 2 hidden layers with an arbitrary number of neurons each
* ReLU activation functions on the hidden layers
* Sigmoid output function
* Adaptive Gradient Algorithm (Adagrad) optimization function

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from bpmll import bp_mll_loss

n = X_train_embeddings.shape[0]
dim_no = X_train_embeddings.shape[1]
class_no = Y_train_clean.shape[1]

print("n:",n,"dim_no:",dim_no,"class_no:",class_no,"\n")

# Simple MLP definition
model = Sequential()
model.add(Dense(128, input_dim=dim_no, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(24, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(class_no, activation='sigmoid', kernel_initializer='glorot_uniform'))
model.summary()
model.compile(loss=bp_mll_loss, optimizer='adagrad', metrics = ['accuracy'])
```

We can train for a few epochs with:
```
model.fit(X_train_embeddings, Y_train_clean, epochs=20, validation_data=(X_dev_embeddings, Y_dev_clean))
```

#### Evaluation of the multi-label classifier

#### Defining a base binary classifier

#### Cross-validation and baseline evaluation for each separate classification task

#### Training multiple binary classifiers

#### Evaluation of the combined binary classifiers

#### Precision, recall and f-score

#### Manual tests with new data and subjective validation

#### Saving the models for use

### The notebook

The development process was done entirely on a [Google Colab notebook](https://colab.research.google.com/) that you can find in this repository.
You may need to adapt the I/O calls as I uses my own Google Drive to store data.

### Practical application

You can find an interactive demonstration here:
https://emotions.aitorperez.com

You can see the process in action on real tweets.

The NLP pipeline runs in Python (tokenizing sentences with NLTK + extracting embeddings with LASER) and has been deployed in a Google Cloud Function.
The trained models were exported and run on the browser with TensorFlowJS.

## Data sources and AI methods
*Where does your data come from? Do you collect it yourself or do you use data collected by someone else?*

### Data

The data used is from the [SemEval2018-Task1](https://competitions.codalab.org/competitions/17751) competition, specifically **the E-c (emotion classification) dataset**.
This dataset has this format:

| ID | Tweet | anger | anticipation | disgust | fear | joy | love | optimism | pessimism | sadness | surprise | trust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017-En-21441 | Do you think humans have the sense for recognizing impending doom? | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

Where each tweet text has been annotated using a binary classification (1 = on average reviewers did infer this emotion from the text and 0 = on average they didn't).

The E-c dataset is divided in train, dev and test datasets.

### Methods

I opted to use a feedforward artificial neural network (ANN) from the start, because I wanted to get first-hand experience building and training one.
Specifically my intention was to use a multilayer perceptron (MLP).
At first, my idea was to apply multi-label classification with a single neural network but after having disappointing results I changed the approach to multiple binary classificators.

## Challenges
*What does your project _not_ solve? Which limitations and ethical considerations should be taken into account when deploying a solution like this?*

### AI challenges

As it stands, this project does a decent job of identifying probable emotions from sentences.
But it sometime fails because it is not taking into consideration the context (the other sentences of the tweet): in some instances it gives hilarious combinations on a single tweet that don't hold upon human review.
Trying to then infer emotion for the full tweet is not feasible with this unless almost each sentence carries similar sentiment (in the demo I average probabilities and show number of occurrences, but it is far from a realistic reasoning).

The training was done with full tweets (in absence of a dataset of sentences) but the real use case is in analyzing sentences and the input features are the LASER embeddings which are intended to be extracted from sentences (not full tweets).

This mismatch probably has made the AI system less accurate.

Also, the dataset is not very extensive (about 6600 tweets).
That hurt at the initial multi-label classification idea and it is still not ideal on the final solution.

### Real-world hurdles

From a practical point of view, I was surprised that the Tensorflow model could be executed in Javascript but the rest of the pipeline being in Pyhton needs a real server and some real disk (the LASER model goes about 200MB, clearly not something that can be ported to a browser).

Not an easy thing to find a way to deploy (for free) Python code and get it running with dependencies like LASER.

### Considerations

*Which ethical considerations should be taken into account when deploying a solution like this?*

## Next steps

To build upon this projec, some next steps that could be taken are:

* Searching for a larger dataset to use in training
* Changing to a dataset more appropriate (one with annotations for sentences)
* Fine-tunning the current neural network design
* Evaluating the current approach with neural networks against other solutions (including simpler ones like Naive Bayes Classifications, Nearest Neighbours, etc.)

As for the applicability, i think that the actual tokenization and prediction is quite fast, so embedding into server or desktop applications would be feasible.
The dependencies (specifically the pre-trained Laser model) are quite heavy so it would be problematic on mobile applications.
An AI like this could be part of a system to watch out for hateful/offensive comments, a system to predict the mental health of people, a way to make reviews richer, or it could also be another input for the priorization of issues on a ticketing platform.

## Acknowledgments

* [Efficient (vectorized) implementation of the BP-MLL loss function](https://github.com/vanHavel/bp-mll-tensorflow)
* [Multi-Label Neural Networks with Applications to Functional Genomics and Text Categorization](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.130.7318&rep=rep1&type=pdf)
* [Multi-Label Classification with Deep Learning](https://machinelearningmastery.com/multi-label-classification-with-deep-learning/)
* [Large-scale Multi-label Text Classification — Revisiting Neural Networks](https://arxiv.org/pdf/1312.5419.pdf)
* [HateMonitors: Language Agnostic Abuse Detection in Social Media](https://arxiv.org/pdf/1909.12642.pdf)
* [Language-Agnostic SEntence Representations (LASER)](https://github.com/facebookresearch/LASER)
* [LASER for NLP tasks](https://www.engati.com/blog/laser-for-nlp-tasks-part-ii)
* [SemEval-2018 Affect in Tweets DIstant Supervision Corpus (SemEval-2018 AIT DISC)](https://competitions.codalab.org/competitions/17751#learn_the_details-datasets)


Images will make your README look nice!
Once you upload an image to your repository, you can link link to it like this (replace the URL with file path, if you've uploaded an image to Github.)
![Cat](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg)

If you need to resize images, you have to use an HTML tag, like this:
<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg" width="300">

This is how you create code examples:
```
def main():
   countries = ['Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden']
   pop = [5615000, 5439000, 324000, 5080000, 9609000]   # not actually needed in this exercise...
   fishers = [1891, 2652, 3800, 11611, 1757]

   totPop = sum(pop)
   totFish = sum(fishers)

   # write your solution here

   for i in range(len(countries)):
      print("%s %.2f%%" % (countries[i], 100.0))    # current just prints 100%

main()
```

