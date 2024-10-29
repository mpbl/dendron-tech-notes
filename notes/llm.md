---
id: 67sop5tguaktyd2gp88fx5i
title: LLM
desc: ''
updated: 1730206416543
created: 1729595367534
---

References : 
- Speech and Language Processing, 3rd ed. Jurafsky Daniel, Martin James H.


## Basics of NLP


### Definitions

- **Words** come from **Corpora**. 
- **lemma / citation form** : set of lexical forms having the same stem
- **wordform** : full inflected or derived form of the word.
- **token** : words or parts of words

When providing a dataset, a *datasheet* or *data statement* should be provided :
- Motivation / Why does it exists ?
- Situation: Which situation the texts written in ?
- Language
- Demographics of authors
- Collection process
- Annotatoin process
- Distribution / Intellectual Property Rights

In order to compare or process texts coherently, text should be normalized, usually using a consistent set of techniques (tokinization, format normalization, sentence segmentation)

### Algorithms 

#### Regular expressions

Since the 1960's, text processing has been done using Regular Expression (regex). While a lot can be achieved using this technique, it forces to hard code the rules.

#### Byte-pair encoding (BPE)

- Creating the token :
    - The token starts as the set of individual characters. Most frequent pairs of token are then iteratively added to the token set.
    - Everytime a new token is created, it replaces the pair of tokens it was built from.

### Concepts

- Sentence segmentation
- Minimum Edit Distance (dynamic programming)
- Alignment (back-trace of the minimum edit distance algorithm)


## N-gram Language Models

### Definitions

- **Language Model**: models assigning probabilites to upcoming words or sequences of words.
- **N-Grams**: sequence of *n* words. $P(w_n | w_{n-N+1:n-1}) \approx  P(x_n | x_{1:n-1})$
- **Bigram**: $P(w_n | w_{n-1}) \approx  P(x_n | x_{1:n-1})$
- **Extrinsic evaluation**: comparing performance of 2 models together
- **Intrinsic evaluation**: evaluation independent of any application
- **Perplexity**: "how much the model is *surprised* by the sentence $W=w_1...w_N$
  - $perplexity(w) = P(w_1w_2..w_N)^{-\frac{1}{N}} = \sqrt[n]{\cfrac{1}{\prod_{i=1}^N P(w_i | w_1...w_{i-1})}}$
  - relation to cross-entropy : $perplexity(W) = 2^{H(W)}$
- **Held-out corpus**: additional training corpus used to compute weight or other metrics

### Algorithms

- **Prediction based on bigrams** : $P(w_{1:n}) = \prod_{k=1}^{n}P(w_k | w_{k-1})$
with a Maximum Likelihood Estimation (MLE), it gives :
  - $P(w_k | w_{k-1}) = \cfrac{Count(w_{n-1}w_n)}{\sum_w Count(w_{n-1}w)} = \cfrac{Count(w_{n-1}w_n)}{Count(w_{n-1})}$
- **Laplace Smoothing**
  - $P_{Laplace}(w_i) = \cfrac{c_i + 1}{N +V}$, with $N$ the number of tokens and $V$ the number of words in the vocabulary
  - adjusted count $c^*_i = (c_i+1)\cfrac{N}{N+V}$
  - discount $d_c = \cfrac{c^*}{c}$
- **Add-k smoothing** : generalization of Laplace smoothing with k instead of 1. Does not usually work well.
- **Backoff** : weighted sum of n-Grams probabilites
  - $\hat{P}(w_n|w_{n-2}w_{n-1}) = \lambda_1 P(w_n) + \lambda_2 P(w_n | w_{n+1}) + \lambda_3 P(w_n|w_{n-2}w_{n-1})$, for a trigram
- **Stupid backoff**: enough for LLMs, no discounting of higher probabilites. Not a real probabiliy distribution.
$$
  S(w_i|w_{i-N+1:i-1}) =
    \begin{cases}
      \cfrac{count(w_{i-N+1:i})}{count(w_{i-N+1:i-1})} &\text{if } count(w_{i-N+1:i}) > 0\\
      \lambda S(w_i | w_{i-N+2:i-1}) &\text{otherwise}
    \end{cases} 
$$
  - base case: $S(w) = \cfrac{count(w)}{N}$
- > **Kneser-Ney** smoothing makes use of the probability of a word being a novel
continuation. The interpolated Kneser-Ney smoothing algorithm mixes a
discounted probability with a lower-order continuation probability.
  
### Concepts

- Dealing with unknown words : add an <UNK> pseudo-word to represent them.
- **Smoothing** or **discounting**: avoiding null probabilites

## Naive Bayes, text classification and sentiment

### Algorithms

#### Naive Bayes

- **Bayesian inference** : Classifier returning the **Maximum Posterior Probability**: $\hat{c} = \underset{c \in C}{\argmax} P(c|d)$, with $C$ the set of classes
  - with Bayes' rule : $\hat{c} = \overbrace{P(d|c)}^{\text{likelihood}}\overbrace{P(c)}^{\text{prior}}$
- **Naive Bayes assumption**: conditional indepedence
  - $c_{NB} = \underset{c in C}{\argmax}\,P(c) \prod_{i \in positions} P(w_i|c)$
  - easier to compute in log space:
  - $c_{NB} = \underset{c in C}{\argmax}\;\log P(c) + \sum_{i \in positions} \log P(w_i|c)$
- **Training NB** : counting frequencies with add-one (Laplace) smoothing 
  - $\hat{P}(w_i|c) = \cfrac{count(w_i, c + 1)}{(\sum_{w \in V} count(w, c)_ + |V|}$

### Concepts

- Text categorization
- Sentiment analysis
- Avoiding _harms_ in classification
  - representational arms (demeans a social group)
  - toxicity detection
  - transparency by using a **model card**
     - training algorithms, parameters
     - training data sources, motivation, and preprocessing
     - evaluation data sources, motivation, and preprocessing
     - intended use and users
     - model performance across different demographic or other groups and environmental situations

## Vector Semantics and Embeddings

### Definitions

- **Distributional hypothesis** : words occurring in _similar contexts_ tend to have _similar meaning_
- **Vector semantics** representation of words as points in a multi-dimensional semantic space, called **embeddings**
  - **static embeddings**
  - **contextualized embeddings** (e.g. BERT)
- **Representation learning** : self-supervised technique in which the learning is made from representations present in the input text.
- **principle of contrast** : a difference in linguistic form is always associated with some difference in meaning.
- **Word Similarity**
- **Word Relatedness** (psychology : association)
- **Semantic field** / **topic models**
