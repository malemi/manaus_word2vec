# word2vec for [manaus](https://github.com/GetJenny/manaus)

This is a word2vec implementation (just a draft, taken from the Udacity course) we used in conjunction to
[manaus](https://github.com/GetJenny/manaus). (manaus is a collection of libraries (in Scala) written for [StarChat](https://github.com/GetJenny/starchat). The main feature is the extraction of keywords, used in [StarChat](https://github.com/GetJenny/starchat) for identifying similar questions.)

## Features

It's been modified in order to work with small dataset, and we have implemented the following features.

### Keywords
Given the fact that logs for chatbots are (at least in our case) not to big, we want to avoid filters in words and use only the most significative ones –hence the keywords.

### Same-sentence words
The second feature of this w2v is that it uses batches of different lenght, using only words from the same sentence. Again, given the limited size of logs we cannot afford to say that the last word in one sentence is neighbour to the first word in the next sentence. We therefore take as context words only the ones in the same sentence.

### Semantic pre-training
The vector used for the initialization of the traditional word2vec are not random. Before doing the actual w2v, we group words at close edit distance, encode them and force the decode to be similar. In practice, we start the actual w2v with words like "dog" and "dogs" having very close vectors.
