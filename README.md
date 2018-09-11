# To be or what to be, that is the question
This is an attempt at this [problem](https://www.hackerrank.com/challenges/to-be-what/problem) listed on HackerRank.

This solution uses an Word2Vec CBOW variation. This was only an attempt that has proven unsuccessful. Please see the [seq2seq solution](https://github.com/bassamriman/ToBeOrWhatToBe-Seq2seq) that has a success rate of 92%.

Shortcoming of this solution:
* Word order is not taken into consideration
* Cannot predict more than one form of verb "be" per sentence

Main libraries used:
* [torchtext](https://github.com/pytorch/text) for data loading, batching and vocabulary labeling
* [pytorch](https://github.com/pytorch/pytorch) for tensor computation with GPU acceleration and deep neural network.
* [nltk](https://www.nltk.org) for training data loading and pre-processing.

##  Usage
Simply run beOrNotToBe-word2vec-trainer after running beOrNotToBe-word2vec-preprocessor (to download training data and pre process it)
