# NamedEntityRecognition
Recognition of Named-Entities in user generated content in Python

In this report, I analyze a neural network model for the Named-Entity Recognition (NER) task, that can be viewed as a classification problem. This task was studied as part of the W-NUT Shared Task in 2017, which challenged participants with unusual and previously unseen entities. I achieve this task by studying two kinds of classification algorithms: a feature-based and a neural network.

## Feature-based classifier
The feature-based classifier relies on the following features:
  - NLTK part-of-speech tags of the current word
  - Scapy part-of-speech tags of the current, and the two previous words
  - are the current and the previous words tagged as a proper noun? (according to Scapy tags)
  - are the current and the previous words in title case?
  - are the current and the previous words in capital case?
  - do the current and the previous words consist of alphabetic characters only?
  - do the current and the previous words consist of numeric characters only?
  - word length of the current and the previous words

I chose to work with a simple logistic regression, as it appears to be the most performant model compared to other models, such as SVM, Gradient Boosting or Naive Bayes ones. I also chose to work with all the training data, despite unbalanced classes. To solve this issue, I have specified different class weights in the logistic regression model, so that smaller classes have a bigger weight in the classification task.

**Additional tricks**
In addition to these features, two methods are used to improve the performance of the classifier. 

First, accoring to Lev Ratinov and Dan Roth (Ratinov  & Roth, 2009), the BILOU tagging scheme is more efficient for NER tasks than the BIO scheme. The BILOU scheme assigns the following tags to words:
  - for a multi-token named entity:
    - B for the first token
    - I for intermediate tokens
    - L for the last token
  - U for a unit named entity
  - O for all other words

Therefore, I have converted BIO labels into BILOU tags, trained a classifier with the BILOU tags, and converted the BILOU predictions into BIO predictions to compute the performance of the system.

Then, I have noticed that BIO predictions are sometimes inconsistent. For exemple, a word can be labeled as I, but the previous word is labeled as O. To fix this problem, I have converted the predicted labels in order that the first token of an entity is labeled as B and the next tokens (if they exist) of the same entity is labeled as I. This method significantly improve the final F1-score of the model.


## Neural Network classifier
In this part, I explore two approaches to improve the performance of a simple neural network. In the first one, I compare the impact on performance of the tagging scheme used to tag named-entities. In the second one, I examine which word embedding representation achieves best results. 

The following code consists of three parts:
  - data processing: data are pre-processed so that they can be used as input of our neural network:
      1. NER labels are converted into the tagging scheme chosen
      2. word tokens and NER labels are encoded as integer values
      3. tokens are converted to lower case
      4. data are transformed into padded sequences of the same length
      5. NER label sequences are encoded with a one-hot scheme and weighted to overcome the imbalanced classesissue
  - neural-network model: 
      1. word embedding vectors are computed according to the method chosen
      2. a bi-LSTM neural network is instantiated and fitted with the training data
  - model evaluation: the model is evaluated with an entity-level F1-score
      1. we remove padding from sentences and convert data into their initial table format 
      2. we convert NER labels into the BIO2 scheme
      3. we evaluate the fitted model on the development dataset

Because of the random initialization of neural networks, I use statistical methods to compare results. I find that we achieve the best results when tagging words with an Inside/Outside scheme for named-entities and using word embeddings pre-trained on the corpus data.

All this work is detailed in my report.

NB: in this notebook, the BIO tagging scheme refers to the BIO2 one, described in the report.
