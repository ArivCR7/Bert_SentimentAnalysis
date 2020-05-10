# SentimentAnalysis

One of the tasks that required manual interpretation is the sentiment analysis where there involves a huge manual effort in analyzing the sentiment of a sentence. The sentence can be a tweet/review/feedback or anything about a company/moview/product/person/party,etc. Designing a system that automatically classifies the sentiment to a positive/negative/neutral would bring down a huge manual effort. 

# Traditional Methods

Sentiment Analysis were tried with some traditional methods that involves finding the positive and negative words from the sentence by comparing each word against a corpus of positive/negative vocabs. And finally classifying the sentence to be positive/negative based on the number of positive/negative words. This method didn't perform good in the scenarios when users post a sarcastic review. 

Another technique that overcame this shortcoming is the neural network based sequence modelling where we train a network in a supervised manner. It required a lot of training for accurate sentence classification. With the advent of transfer learning in NLP, it became very easy to use a pretrained model that learnt the better language representation to perform a particular task.

# SOTA - Transfer Learning in NLP
In this repo, we use BERT(Bidirectional Encoder Representational Transformers) to perform sentiment analysis task. To understand BERT better, refer the link below:

https://www.youtube.com/watch?v=x66kkDnbzi4&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6

# Reference
https://www.youtube.com/watch?v=hinZO--TEk4


