# Multi-languange Twitter sentiment analysis.

Multi-languange affect analysis using neural networks and zero-shot cross-lingual transfer learning.
Final project for the Building AI course.

*December 2020*

## Summary

TL;DR: this project builds an **AI that predicts 11 emotions from text sentences in 93 languages** and you can try it on real tweets going to https://emotions.aitorperez.com

![Desktop screenshot](/docs/emotions.png)

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

### Development process

#### Dependencies

```python
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

First, we load the data to numpy arrays from the txt files splitting into input arrays and output arrays by columns:

```python
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

```python
count = 0
for element in Y_subset:
  if np.sum(element) == 0:
    count +=1
print(count)
```

On the training subset we find these tweets without emotion:

```python
204
```

To remove this instances, we can use the following:

```python
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

```python
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

```python
from laserembeddings import Laser
laser = Laser()
X_train_embeddings = laser.embed_sentences(X_train_clean,lang='en')
```

This results in a matrix with this shape for the trainig dataset:

```python
(6634, 1024)
```

#### Training the multi-label classifier

We first define the following MLP using as a template the example one from the BP-MLL implementation.
* 2 hidden layers with an arbitrary number of neurons each
* ReLU activation functions on the hidden layers
* Sigmoid output function
* Adaptive Gradient Algorithm (Adagrad) optimization function

```python
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

```python
model.fit(X_train_embeddings, Y_train_clean, epochs=20, validation_data=(X_dev_embeddings, Y_dev_clean))
```

#### Evaluation of the multi-label classifier

The training results show below average results (far worse than a coin toss):

```python
Epoch 20/20
208/208 [==============================] - 3s 13ms/step - loss: 0.9654 - accuracy: 0.3850 - val_loss: 0.9643 - val_accuracy: 0.3612
```

The trained model can be further evaluated with the test dataset:

```python
results = model.evaluate(X_dev_embeddings, Y_dev_clean, batch_size=128, verbose=0)
print("test loss, test acc:", results)
```

With results in the same level of failure:

```python
test loss, test acc: [0.9661824107170105, 0.34579145908355713]
```

Some experimentation

With one hidden layer:

| Configuration | Test Loss | Test Accuracy |
| --- | --- | --- |
| N = 2048 | 0.9317 | 0.3304 |
| N = 1024 | 0.9345 | 0.3433 |
| N = 128 | 0.9518 | 0.3304 |
| N = 64 | 0.9592 | 0.3433 |

With two hidden layers:

| Configuration | Test Loss | Test Accuracy |
| --- | --- | --- |
| N = 1024, 22 | 0.9527 | 0.0616 |
| N = 256, 32 | 0.9391 | 0.3304 |
| N = 256, 22 | 0.9377 | 0.3304 |
| N = 128, 32 | 0.9457 | 0.0616 |
| N = 128, 22 | 0.9278 | 0.3304 |
| N = 64, 22 | 0.9531 | 0.0616 |

It appears that the optimization plateaus at around 0.33 accuracy, whatever the configuration (and an even worse local minimum that gives 0.06 accuracy).
From the previous analysis of the corpus it was clear that the training subset was small and not balanced, and that is probably the root cause of these observations.

Taking from each experiment the best simple configuration and introducing dropout:

| Configuration | Test Loss | Test Accuracy |
| --- | --- | --- |
| N = 128, d = 0.2, N = 22 | 0.956 | 0.3304 | 
| N = 128, d = 0.3, N = 22 | 0.9668 | 0.0616 | 
| N = 128, d = 0.4, N = 22 | 0.9499 | 0.606 | 
| N = 64, d = 0.2 | 0.9634 | 0.3458 | 
| N = 64, d = 0.3 | 0.9466 | 0.2999 | 
| N = 64, d = 0.4 | 0.9585 | 0.1941 | 

In summary: the simpler, the better (though bad anyways).

When testing concrete tweets, the results were rather weird:

```python
Distance yourself once stretched by your friends impose! serious loveyou notseriously
[0.6  0.47 0.58 0.48 0.62 0.35 0.55 0.45 0.53 0.39 0.36]
joy with prob 0.6159021854400635
all emotions: anger, disgust, joy, optimism, sadness 

Be happy. Be confident. Be kind.\n\n KissablesLoveSMShopmag\nAllOutDenimFor KISSMARC
[0.59 0.48 0.57 0.48 0.61 0.36 0.54 0.46 0.54 0.39 0.38]
joy with prob 0.6131008267402649
all emotions: anger, disgust, joy, optimism, sadness 

This time in 2 weeks I will be 30... 😥
[0.59 0.47 0.58 0.48 0.62 0.35 0.54 0.45 0.53 0.4  0.37]
joy with prob 0.6196826100349426
all emotions: anger, disgust, joy, optimism, sadness
```

In light of these results it seems clear that the intention of building a single multi-label classificator with this dataset is way too ambitious.
Instead, I decided to break the task into single classifiers (emotion / no emotion) for each of the 11 emotions with the clear drawback that in production this would be 11 times more expensive to compute (and if computing a single production would take too long, computing 11 would be impractical).

#### Defining a base binary classifier

We'll need to generate a model for each emotion.
The definition is encapsulated into a function like this:

```python
# baseline model
def create_baseline(number_of_classes=class_no):
  # create model
  model = Sequential()
  model.add(Dense(128, input_dim=dim_no, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(number_of_classes, activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
```

This time, each classifier uses a well-known configuration for the binary classification task:
* ReLU activation functions
* Sigmoid output function
* Binary cross-entropy loss function
* Adam optimization function

#### Cross-validation and baseline calculation for each separate classification task

Before training in full the 11 classifiers (which can take some time) I tried to predict if this model would be able to generalize well.
Doing a little research, stratified k-fold cross-validation is what's called for here.

As per wikipedia, cross-validations are:

> Model validation techniques for assessing how the results of a statistical analysis will generalize to an independent data set.

> In stratified k-fold cross-validation, the folds are selected so that the mean response value is approximately equal in all the folds. In the case of a dichotomous classification, this means that each fold contains roughly the same proportions of the two types of class labels.

The sklearn package allows to easily do that:

```python
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

# evaluate model with stratified datasets
stats = []
for i in range(11):
  ordinal = i
  X_input = X_train_embeddings
  Y_output = Y_train_clean[:,ordinal]

  estimator = KerasClassifier(build_fn=create_baseline, number_of_classes=1, epochs=5, batch_size=128, verbose=0)
  kfold = StratifiedKFold(n_splits=4, shuffle=True)
  results = cross_val_score(estimator, X_input, Y_output, cv=kfold)
  print(labels[i], "baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
  stats.append(results.mean())

print("average: %.2f%% (%.2f%%)" % (np.mean(stats)*100, np.std(stats)*100))
```

The results on the final model were:

| Emotion | Mean (score) | Standard deviation (score) |
| --- | --- | --- |
| anger | 74.62% | 1.95% |
| anticipation | 85.26% | 0.03% |
| disgust | 75.03% | 1.50% |
| fear | 81.40% | 0.12% |
| joy | 79.38% | 0.79% |
| love | 89.45% | 0.00% |
| optimism | 76.21% | 0.74% |
| pessimism | 88.02% | 0.02% |
| sadness | 72.28% | 0.43% |
| suprise | 94.56% | 0.03% |
| trust | 94.62% | 0.03% |

On average (if emotions were independent) the model gets a mean score of 82.68% with a std of 7.89%
These are far from perfect, but now good enough for a first project like this.

#### Training multiple binary classifiers

Using the model function, we can automate the training of the 11 models:

```python
#fit the model to all dimensions
models = []
for i in range(class_no):
  model = create_baseline(number_of_classes=1)
  Y_train_split = Y_train_clean[:,i]
  Y_dev_split = Y_dev_clean[:,i]
  print("processing model for",labels[i])
  model.fit(X_train_embeddings, Y_train_split, epochs=10, validation_data=(X_dev_embeddings, Y_dev_split), verbose=0)
  models.append(model)
```

#### Evaluation of the combined binary classifiers

After training, the loss and accuracy values on the validation subset were:

```python
anger val loss, val acc: [0.4944448471069336, 0.7568807601928711]
anticipation val loss, val acc: [0.39562666416168213, 0.8589449524879456]
disgust val loss, val acc: [0.4880869388580322, 0.7568807601928711]
fear val loss, val accl: [0.25919538736343384, 0.9036697149276733]
joy val loss, val acc: [0.44426000118255615, 0.78899085521698]
love val loss, val acc: [0.29084280133247375, 0.8715596199035645]
optimism val loss, val acc: [0.4670793414115906, 0.7844036817550659]
pessimism val loss, val acc: [0.3221796154975891, 0.8910550475120544]
sadness val loss, val acc: [0.4782714545726776, 0.7683486342430115]
surprise val loss, val accl: [0.14201349020004272, 0.9598624110221863]
trust val loss, val acc: [0.18342220783233643, 0.9506880640983582]
```

Way better than before!

#### Precision, recall and f-score

* *Precision* quantifies the number of positive class predictions that actually belong to the positive class.
* *Recall* quantifies the number of positive class predictions made out of all positive examples in the dataset.
* *F-Score* provides a single score that balances both the concerns of precision and recall in one number.

Again using sklearn this is easy to calculate:

```python
# Calculate F1 score for all dimensions

for i in range(class_no):
  true = Y_test[:,i]
  prediction = models[i].predict(X_test_embeddings)

  # Normalize output
  from sklearn.preprocessing import Binarizer
  transformer = Binarizer(threshold=0.5).fit(prediction)  # fit does nothing.
  prediction = transformer.transform(prediction)

  #calc
  from sklearn.metrics import f1_score, recall_score, precision_score

  recall = recall_score(y_true=true, y_pred=prediction, average='weighted')
  precision = precision_score(y_true=true, y_pred=prediction, average='weighted')
  score = f1_score(y_true=true, y_pred=prediction, average='weighted')

  print("%.2f %.2f %.2f" % (recall, precision, score), labels[i])
```

On the final model this looked like:

| emotion | recall | precision | f-score |
| --- | --- | --- | --- |
| anger | 80% | 80% | 80% |
| anticipation | 87% | 81% | 81% |
| disgust | 77% | 77% | 77% |
| fear | 90% | 89% | 88% |
| joy | 82% | 82% | 82% |
| love | 88% | 88% | 88% |
| optimism | 77% | 76% | 76% |
| pessimism | 89% | 85% | 84% |
| sadness | 78% | 77% | 78% |
| suprise | 95% | 90% | 92% |
| trust | 95% | 91% | 93% |

#### Manual tests with new data and subjective validation

A quick test code simulating the real-world usage would be:

```python
import re
regex = r"[.]{2,3}"
subst = "."
regexed = re.sub(regex, subst, INPUT_TEXT)

from nltk import tokenize
sentences = tokenize.sent_tokenize(regexed)
embeddings = laser.embed_sentences(sentences,lang='en')
print(sentences)
print(embeddings.shape)

from sklearn.preprocessing import Binarizer
threshold = 0.5
X=[[0]]
transformer = Binarizer(threshold=threshold).fit(X)  # fit does nothing.

for i, sentence in enumerate(sentences):
  print("\n", sentence)
  single_embedding = np.array([embeddings[i]])
  for j, model in enumerate(models):
    prediction = model.predict(single_embedding)
    # Normalize output    
    normalized = transformer.transform(prediction)
    result = bool(normalized[0,0])
    if result: print("guessed", labels[j], "with prob", prediction[0,0])
```

And a sample output (using a spanish tweet):

```python
['Estando discapacitado mi ex pareja después de 10 años de relación me dejó y me culpó de mi salud.', 'Tuve que cerrar mi empresa y tuve que irme a vivir a casa de mi padre viudo y mayor para que me cuidara.', 'Todos tenemos la fuerza para seguir adelante, solo hay que encontrarla.']
(3, 1024)

Estando discapacitado mi ex pareja después de 10 años de relación me dejó y me culpó de mi salud.
guessed anger with prob 0.60721034
guessed sadness with prob 0.72548664

Tuve que cerrar mi empresa y tuve que irme a vivir a casa de mi padre viudo y mayor para que me cuidara.

Todos tenemos la fuerza para seguir adelante, solo hay que encontrarla.
guessed optimism with prob 0.7915448
```

Which looks not bad at all!

#### Saving the models for use

Finally, to export the models and weights to be able to use them later:

```python
for i,model in enumerate(models):
  # serialize model to JSON
  model_json = model.to_json()
  with open("/content/drive/MyDrive/Machine Learning/Datasets/SemEval-2018 Affect in Tweets E-c/"+labels[i]+"-model.json", "w") as json_file:
      json_file.write(model_json)

  # serialize weights to HDF5
  model_weights = model.save_weights("/content/drive/MyDrive/Machine Learning/Datasets/SemEval-2018 Affect in Tweets E-c/"+labels[i]+"-weights.h5")
  
  # save as TensorFlow.js Layers format
  tfjs.converters.save_keras_model(model, "/content/drive/MyDrive/Machine Learning/Datasets/SemEval-2018 Affect in Tweets E-c/"+labels[i])
```

This saves the model (as a JSON file) and its weights (as a h5 file), which can be imported back again in python.
It also saves a bundled version (a JSON file + a number of bin files) to a folder, which can be used in javascript via TensorflowJS.

### The notebook

This development process was done entirely on a [Google Colab notebook](https://colab.research.google.com/) that you can find in this repository.
You may need to adapt the I/O calls as it uses paths from my own Google Drive to store data.

### Practical application

You can find an interactive demonstration at https://emotions.aitorperez.com where you can see the process in action on real tweets.

* The NLP pipeline runs in Python (tokenizing sentences with NLTK + extracting embeddings with LASER) and has been deployed in a Google Cloud Function.
* The trained models run on the browser with TensorFlowJS.
* The frontend was developed with React and Material-UI and is deployed at Netlify

![Mobile screenshot](/docs/mobile.png)

## Data sources and AI methods

### Data

The data used is from the [SemEval2018-Task1](https://competitions.codalab.org/competitions/17751) competition, specifically **the E-c (emotion classification) dataset**.
This dataset has this format:

| ID | Tweet | anger | anticipation | disgust | fear | joy | love | optimism | pessimism | sadness | surprise | trust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017-En-21441 | Do you think humans have the sense for recognizing impending doom? | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

Where each tweet text has been annotated using a binary classification (1 = on average reviewers did infer this emotion from the text and 0 = on average they didn't).
The annotations were crowdsourced and all the metodology is documented in a paper.

The E-c dataset is divided in train (6838), dev (886) and test (3256) datasets.

### Methods

I opted to use a feedforward artificial neural network (ANN) from the start, because I wanted to get first-hand experience building and training one.
Specifically my intention was to use a multilayer perceptron (MLP).
At first, my idea was to apply multi-label classification with a single neural network but after having disappointing results I changed the approach to multiple binary classificators.

## Challenges

### AI challenges

As it stands, this project does a decent job of identifying probable emotions from sentences.
But it sometime fails because it is not taking into consideration the context (the other sentences of the tweet): in some instances it gives hilarious combinations on a single tweet that don't hold upon human review.
Trying to then infer emotion for the full tweet is not feasible with this unless almost each sentence carries similar sentiment (in the demo I average probabilities and show number of occurrences, but it is far from a realistic reasoning).

Another way it fails is because it is unaware of sarcasm an irony (which Twitter is full of).

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

To build upon this project, some next steps that could be taken are:

* Searching for a larger dataset to use in training, or
* Replacing it with a dataset more appropriate (one with annotations for sentences, for example)
* Fine-tunning the current neural network design
* Evaluating other solutions (including simpler ones like Naive Bayes Classifications, Nearest Neighbours, etc.)
* Introduce sarcasm, irony into the datasets

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

Libraries:

* [nltk: a natural language platform in python](https://www.nltk.org/)
* [laserembeddings: a pip-packaged, production-ready port of LASER](https://github.com/yannvgn/laserembeddings)
* [keras: a neural network library for python](https://keras.io/)
* [tensorflow: a platform for machine learning](https://www.tensorflow.org/)
* [scikit-learn: a machine learning and statistical library for python](https://scikit-learn.org/stable/)
