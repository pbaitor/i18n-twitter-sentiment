# Multi-languange sentiment analysis using neural networks and zero shot cross lingual transfer learning.

Final project for the Building AI course

## Summary

Describe briefly in 2-3 sentences what your project is about. About 250 characters is a nice length! 


## Background

Which problems does your idea solve? How common or frequent is this problem? What is your personal motivation? Why is this topic important or interesting?

This is how you make a list, if you need one:
* problem 1
* problem 2
* etc.

* Using a publicly available dataset in English for training and testing
* Training a Neural Network for multi-label classification using an adapted method (BP-MLL)
* Predicting labels for different languages

## How is it used?

Describe the process of using the solution. In what kind situations is the solution needed (environment, time, etc.)? Who are the users, what kinds of needs should be taken into account?

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


## Data sources and AI methods
Where does your data come from? Do you collect it yourself or do you use data collected by someone else?
If you need to use links, here's an example:
[Twitter API](https://developer.twitter.com/en/docs)

The data used is from the [SemEval2018-Task1](https://competitions.codalab.org/competitions/17751) competition, specifically **the E-c (emotion classification) dataset**.
This dataset has this format:

| Field      | Description | Example     |
| ----------- | ----------- | ----------- |
| ID      | Text       | 2017-En-21441       |
| Tweet   | Text        | Do you think humans have the sense for recognizing impending doom?       |
| anger   | binary value        | 0       |
| anticipation   | binary value        | 1       |
| disgust   | binary value        | 0       |
| fear   | binary value        | 0       |
| joy   | binary value        | 0       |
| love   | binary value        | 0       |
| optimism   | binary value        | 0       |
| pessimism   | binary value        | 1       |
| sadness   | binary value        | 0       |
| surprise   | binary value        | 0       |
| trust   | binary value        | 0       |

| ID | Tweet | anger | anticipation | disgust | fear | joy | love | optimism | pessimism | sadness | surprise | trust |
| 2017-En-21441 | Do you think humans have the sense for recognizing impending doom? | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

Where each tweet text has been annotated using a binary classification (1 = on average reviewers did infer this emotion from the text and 0 = on average they didn't).

## Challenges

What does your project _not_ solve? Which limitations and ethical considerations should be taken into account when deploying a solution like this?

## Next steps

Some next steps could be:

* Trying to find a larger dataset to use in training
* Finding a more appropriate dataset (one with annotations for sentences)
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

* list here the sources of inspiration 
* do not use code, images, data etc. from others without permission
* when you have permission to use other people's materials, always mention the original creator and the open source / Creative Commons licence they've used
  <br>For example: [Sleeping Cat on Her Back by Umberto Salvagnin](https://commons.wikimedia.org/wiki/File:Sleeping_cat_on_her_back.jpg#filelinks) / [CC BY 2.0](https://creativecommons.org/licenses/by/2.0)
* etc
