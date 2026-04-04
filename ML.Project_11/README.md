# Attention

Summary: This project introduces the attention mechanism.

💡 [Tap here](https://new.oprosso.net/p/4cb31ec3f47a4596bc758ea1861fb624) **to leave your feedback on the project**. It's anonymous and will help our team make your educational experience better. We recommend completing the survey immediately after the project.

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [Machine Translation before attention](#machine-translation-before-attention) \
    2.2. [How attention works](#how-attention-works) \
    2.3. [Positional encoding](#positional-encoding) \
    2.4. [Attention for sequences with temporal dependency](#attention-for-sequences-with-temporal-dependency)
3. [Chapter III. Goal](#chapter-iii-goal)
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task)
6. [Chapter VI. Bonus part](#chapter-vi-bonus-part)

## Chapter I. Preamble

In the last project we got acquainted with recurrent neural networks — the architecture that allows to process sequential data. And earlier, whenever you need to solve any text processing task, you meet this architecture as one of the approaches. But in 2017, Google had published an article titled "Attention is all you need", where the authors proposed a new architecture for working with sequences and a particularly new model for the machine translation task called Transformer. But the main change was the Attention mechanism inside the model, and it was this that allowed the approach to leapfrog [the existing State of the Art](https://paperswithcode.com/sota/machine-translation-on-wmt2014-english-german) (State of the Art or SOTA is the leading approach for a particular task). The community quickly picked up the idea and tested it in various tasks. And now many well-known models use this architecture internally (BERT, GPT, Dall-e, AlphaFold, etc.). Let's try to understand how this model works and why it works so well.

## Chapter II. Introduction

### Machine Translation before attention

The first task to which the attention mechanism was applied was machine translation, i.e. translating a sentence in one language into another. And before we get into the topic of attention, we would like to analyze how this problem was solved using neural networks before the article was published. Most approaches were based on the Encoder-Decoder architecture, where both the Encoder and the Decoder were separate RNN models. However, these models were trained together, and the last hidden state of the Encoder was fed into the input of the Decoder as the initial state of the hidden state.

![Machine Translation before attention](misc/images/machine_translation_before_attention.png)

But there is a downside to this approach. Understanding exactly what that drawback is the part of your task.

### How attention works

When we translate a phrase within a sentence into another language, in most cases we don't need to see the entire sentence. The translation of a particular phrase is usually affected by only a few words, as shown in the figure below:

![How attention works](misc/images/how_attention_works.png)

So the question is: can we train the neural network to make such connections on its own? In other words, so that the network defines which word is strongly associated with the translation and which is not. If we take the already known RNN, then there is no direct answer to these questions in it. To achieve this in the Transformer model, the authors proposed the following next steps (below, some layers with transformations are omitted for simplicity).

1. For each word in the original sentence and in the translation, we form a vector of fixed size and call them "key" and "query", respectively.

    ![translation](misc/images/translation.png)

2. Between each word in the original sentence and in the translation, we consider the similarity by taking the scalar product of their "key" and "query" representations (the larger the scalar product, the more similar the words are).

    ![translation 2](misc/images/translation_2.png)

3. In parallel, for each word of the original sentence, we also form the "value" vector.

    ![Value vector](misc/images/value_vector.png)

4. And finally, to describe the hidden state of the translated word, we take the weighted sum of the "value" vectors of the original sentence, where the weight is the similarity of the words.

    ![Hidden state](misc/images/hidden_state.png)

Thus, the neural network directly considers the relationship between words (multiplying their Q and K representations), and based on these values, forms a hidden state vector for each word of the translated sentence. By following these steps for each word in the translation, we will get exactly the same output from the network as in RNN:

![RNN](misc/images/rnn.png)

However, we noticed above that we left out some layers in the description. In the original article, the attention mechanism is described as follows:

![Scaled Dot-Product Attention](misc/images/scaled_dot_product_attention.png)

```math
\text {Attention}(Q, K, V) = \text {softmax}(\frac {QK^T} {\sqrt{d_k}})V
```

where:

- Scale layer — simple division of the scalar product by a coefficient that depends on the size of the embedding. This trick allows the softmax layer to be more numerically stable: "We suspect that for large values of dk, the dot products become large, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by √(d_k)".
- Mask layer we will consider further.
- The softmax layer is needed to transform the scaled dot products into probabilities that sum to 1.

However, the authors did not limit themselves to adding one attention layer, but decided to use several such layers in parallel. They called it multi-head attention:

![Multi-head attention](misc/images/multi_head_attention.png)

```math
\begin {align*}
\text {MultiHead}(Q, K, V) &= \text {Concat}(\text{head}_1, \dots, \text {head}_n)W^O \\
\text {where } \text{head}_i &= \text {Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end {align*}
```

And this approach works better than using a single attention layer (even if the models have the same number of parameters). This is because each head can look at the same data from different angles. For example, the first head will determine the gender combination of the words, the second will determine the tense combination, and so on.

It is also important to note that attention can be used not only for 2 different sequences, but also when we are dealing with one sequence. Then the model learns how the elements of the same sequence are related to each other. This is called self-attention:

![transformer-novel-neural-network](misc/images/transformer-novel-neural-network.png)
![transformer-novel-neural-network](misc/images/transformer-novel-neural-network_2.png)

### Positional encoding

Having learned how the attention mechanism works, we can conclude that such a construction allows to solve the main problem of RNN — the loss of connection between distant words. However, there is also a disadvantage: currently we do not take into account the order of words. If we want to translate the text (and we don't have an RNN that has pre-processed it) and feed the model two different sentences: "growth has slowed down" and "down growth has slowed", then the Decoder that works on the Attention mechanism will not see the difference. The set of words remains the same, their embeddings are the same, {Q, V, K} will be the same, and therefore the attention weights and the hidden state will not change. Fortunately, this problem is very easily solved by changing the embeddings that we give as input to compute {Q, V, K}. In this project you will learn about 2 possible solutions:

1. using sine and cosine functions of different frequencies;
2. using trainable position weights. \
    How they are arranged is up to you.

### Attention for sequences with temporal dependency

Let's go back to the Mask layer we observed at the beginning. Imagine a situation where we use a model with a self-attention mechanism inside to predict from a user's clickstream when they will leave the site. And let the clickstream contain the following events:

*visit page A, add item to cart, visit page B, visit cart, visit payment page, afk for 2 min, visit account page, exit*.

If we use self-attention as described above, then when calculating the hidden state for the "visit cart" event, we will use all subsequent events. But then the model will be using information from the future to describe the current event. This situation is fraught with the problem that the model may learn patterns that it cannot observe on real data, because in this case it cannot look into the future. This is called data leakage. It usually leads to overly optimistic results during the model building phase, followed by the unpleasant surprise of poor results after the predictive model is implemented and tested on new data. To avoid this in general, always use validation and test data, and split them up similarly to how the model will work in production.

To avoid this problem for the attention mechanism — we will just assign -□inf for all weights "from the future", then after the softmax layer they will turn to 0:

<table>
  <tr>
    <th>key <br> query</th>
    <th>visit page A</th>
    <th>add to cart item</th>
    <th>visit page B</th>
    <th>visit cart</th>
    <th>visit payment page</th>
  </tr>
  <tr>
    <th>visit page A</th>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
  </tr>
  <tr>
    <th>add to cart item</th>
    <td>...</td>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
  </tr>
  <tr>
    <th>visit page B</th>
    <td>...</td>
    <td>...</td>
    <td>-inf</td>
    <td>-inf</td>
    <td>-inf</td>
  </tr>
  <tr>
    <th>visit cart</th>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>-inf</td>
    <td>-inf</td>
  </tr>
  <tr>
    <th>visit payment page</th>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>-inf</td>
  </tr>
</table>

## Chapter III. Goal

The goal of this task is to get a deep understanding of how to build models with the Attention mechanism.

## Chapter IV. Instructions

- This project will be evaluated by humans only. You are free to organize and name your files as you wish.
- We will use Python 3 as the only correct version of Python.
- For training deep learning algorithms you can try Google Colab. It offers free kernels (Runtime) with GPU, which is faster than CPU for such tasks. The standard is not applied to this project. However, you are asked to be clear and structured in your source code design.
- Store the datasets in the subfolder data

## Chapter V. Task

In this project you will work with a dataset of names and their Russian translations. We will build different machine translation models using different architectures.

You can use any framework you like: PyTorch, TensorFlow, Keras, etc.

### Dataset

You will be working with the dataset generated from an [open dataset of baby names](https://data.world/alexandra/baby-names). The original dataset consists of 6782 records. Each record contains the baby's name and gender. The extended version also includes a Russian translation of all names. The translation was done by Google Translate and then corrected manually in some cases.

### Task

1. Give the answer to the question from the introduction \
    a. TODO.

2. Data preparation \
    a. Download the dataset from the course files. \
    b. Create 2 dictionaries for English letters and Russian letters. Do not forget special tokens. \
    c. Encode names. \
    d. Divide the dataset into training, valid and test parts (take 80-10-10%). Be sure to use random_state.

3. RNN Encoder \
    a. Train a 1-layer RNN model (GRU or LSTM) to encode the English name. Use built-in class of Neural Network Library. \
    b. Add early stopping by validation metric to training procedure. \
    c. Generate 10 names using the trained model.

4. RNN Machine Translation \
    a. As Encoder, take the fitted RNN model from the 3rd point. \
    b. Implement RNN decoder to produce translation. Use built-in class of neural network library. \
    c. Train the network using loss for next letter prediction. Use early stopping for validation metric. Do not train encoder parameters, train Decoder only. \
    d. Implement a method to translate the name. Method should not be stochastic — choose next letter with argmax. \
    e. Evaluation: 

    - i. Plot loss convergence on train and valid data. 
    - ii. Print translation for 5 names from train and 5 names from valid part of dataset every n epochs. Are they realistic? 
    - iii. Compute perplexity on test set.

5. Attention Machine Translation \
    a. As Encoder, take the fitted RNN model from the 3rd point. \
    b. Implement decoder with attention mechanism to produce translation. Do not use the built-in class of the neural network library for Attention. Compute attention between encoder output and decoder input. \
    c. Train the network using loss for next letter prediction. Use early stopping for the validation metric. Do not train Encoder parameters, only Decoder. \
    d. Repeat 4.e for this model. \
    e. Compare the results. What can you learn from these results?

6. Positional Encoding \
    a. Implement position encoding using sine and cosine: 
     
    - i. Train a separate RNN Encoder with this type of positional coding. 
    - ii. Train RNN Decoder for Machine Translation with this type of positional encoding. 
    - iii. Train Attention Decoder for Machine Translation with this type of positional encoding. 
    - iv. Evaluate models and compare results. Make local inferences about the results. 

    b. Implement position encoding with trainable weights: 
    
    - i. Repeat steps 6.a.i–6.a.iii for this positional encoding. 
    - ii. Visualize the learned positional encoding. 

    c. What are the advantages and disadvantages of using sine and cosine positional encoding instead of trainable weight positional encoding? Give the answer.

7. Multi-head Attention Machine Translation \
    a. Implement Multi-head attention. Use n_heads = 3. \
    b. Select the name and add the visualization of the attention matrix within the training process. Display the plot every n epochs. \
    c. Train the Machine Translation with this block. \
    d. Evaluate the model (repeat 4.e). \
    e. One of the heads should be almost diagonal with a shift of 1 block. Explain why there is this shift. Example:

![Example](misc/images/example.png)

### Submission

Save your Multi-Head Attention Machine Translation model in pickle format. Your peer will load it and use it to make predictions again for the test dataset (remember to use the same random state). The predictions should be saved in a file named `test_predictions.csv`.

Your repository should contain one or more notebooks with your solutions.

## Chapter VI. Bonus Part

- Repetition of the [Transformer Architecture](https://arxiv.org/pdf/1706.03762.pdf) for the name translation task.
- Transform names and translations into n-gram vectors. Implement and measure the BLEU metric.

>Please leave feedback on the project in the [feedback form.](https://forms.yandex.ru/cloud/646b47cd02848f2ef1031b24/) 
