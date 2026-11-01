
## CS 336- lecture-1 
Training your models is sometimes needed, Fundamental researarch


#### Flops and Layers
SLM v/s LLM (keep the diagram)
Flops spent -> Majority of time is the FFN 

Scale brings improvements, zero shot and few short

Algorithms that scale are what matter



.... Need to fill here 


#### Tokenization 
Tokens to numbers - > encode and decode


Compression Ratio - No. of Bytes per token
Larger compression solves better attention; shorter sentence as it scales quadratic,

Increase vocab size, better compression ratio 

Then you have more sparsity, that becomes problem more memory is needed


Ways to create tokenizer
Unicode string, each char is a token. 
150k unicode chars -> It is huge vocab
Many chars are rare, not needed



UTF-8 encoding, string to bytes
Byte tokenizer (andrej karpathy)
all numbers between 0-255, compression ratio is 1 (not great)


Word Tokenizer -> chunk it up by special character, based on the dataset. Bag of Words starategy
Huge vocab, not handle missing cases, 



Byte Pair Encoding -> Check this video
Initally for compression, Train tokenizer on raw text for that dataset.
Rare words are multiple token, common words single token
Merge successive pairs of adjacent tokens, that are most common



## CS 336 Lecture -3 - Architectures

