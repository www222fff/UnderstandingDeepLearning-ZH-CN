Chapter 10 introduced convolutional networks, which are specialized for processing data that lie on a regular grid. They are particularly suited to processing images, which have a very large number of input variables, precluding the use of fully connected networks. Each layer of a convolutional network employs parameter sharing so that local image patches are processed similarly at every position in the image.

This chapter introduces transformers. These were initially targeted at natural lan- guage processing (NLP) problems, where the network input is a series of high-dimensional embeddings representing words or word fragments. Language datasets share some of the characteristics of image data. The number of input variables can be very large, and the statistics are similar at every position; it’s not sensible to re-learn the meaning of the word dog at every possible position in a body of text. However, language datasets have the complication that text sequences vary in length, and unlike images, there is no easy way to resize them.

## 12.1 Processing text data

To motivate the transformer, consider the following passage:

```
The restaurant refused to serve me a ham sandwich because it only cooks vegetarian food. In the end, they just gave me two slices of bread. Their ambiance was just as good as the food and service.
```

The goal is to design a network to process this text into a representation suitable for downstream tasks. For example, it might be used to classify the review as positive or negative or to answer questions such as “Does the restaurant serve steak?”.

We can make three immediate observations. First, the encoded input can be surpris- ingly large. In this case, each of the 37 words might be represented by an embedding vector of length 1024, so the encoded input would be of length 37 × 1024 = 37888 even for this small passage. A more realistically sized body of text might have hundreds or even thousands of words, so fully connected neural networks are impractical.

Second, one of the defining characteristics of NLP problems is that each input (one or more sentences) is of a different length; hence, it’s not even obvious how to apply a fully connected network. These observations suggest that the network should share parameters across words at different input positions, similarly to how convolutional networks share parameters across different image positions.

Third, language is ambiguous; it is unclear from the syntax alone that the pronoun it refers to the restaurant and not to the ham sandwich. To understand the text, the word it should somehow be connected to the word restaurant. In the parlance of transformers, the former word should pay attention to the latter. This implies that there must be connections between the words and that the strength of these connections will depend on the words themselves. Moreover, these connections need to extend across large text spans. For example, the word their in the last sentence also refers to the restaurant.

## 12.2 Dot-product self-attention
The previous section argued that a model for processing text will (i) use parameter sharing to cope with long input passages of differing lengths and (ii) contain connections between word representations that depend on the words themselves. The transformer acquires both properties by using *dot-product self-attention*.

A standard neural network layer $f[x]$, takes a $D \times 1$ input $x$ and applies a linear transformation followed by an activation function like a ReLU, so:

$$f[x] = ReLU[\beta + \Omega x],$$

where $\beta$ contains the biases, and $\Omega$ contains the weights.

A self-attention block $sa[\cdot]$ takes $N$ inputs $x_1, \ldots, x_N$, each of dimension $D \times 1$, and returns $N$ output vectors of the same size. In the context of NLP, each input represents a word or word fragment. First, a set of *values* are computed for each input:

$$v_m = \beta_v + \Omega_vx_m,$$

where $\beta_v \in \mathbb{R}^D$ and $\Omega_v \in \mathbb{R}^{D \times D}$ represent biases and weights, respectively.

Then the $n^{th}$ output $sa_n[x_1, \ldots, x_N]$ is a weighted sum of all the values $v_1, \ldots, v_N$:

$$sa_n[x_1, \ldots, x_N] = \sum_{m=1}^N \alpha[x_m, x_n]v_m.$$

The scalar weight $\alpha[x_m, x_n]$ is the *attention* that the $n^{th}$ output pays to input $x_m$. The $N$ weights $\alpha_{p, x_n}$ are non-negative and sum to one. Hence, self-attention can be thought of as *routing* the values in different proportions to create each output (figure 12.1).

The following sections examine dot-product self-attention in more detail. First, we consider the computation of the values and their subsequent weighting (equation 12.3). Then we describe how to compute the attention weights $\alpha[x_m, x_n]$ themselves.

### 12.2.1 Computing and weighting values
Equation 12.2 shows that the same weights $\Omega_v \in \mathbb{R}^{D \times D}$ and biases $\beta_v \in \mathbb{R}^D$ are applied to each input $x_n \in \mathbb{R}^D$. This computation scales linearly with the sequence length $N$, so it requires fewer parameters than a fully connected network relating all $DN$ inputs to all $DN$ outputs. The value computation can be viewed as a sparse matrix operation with shared parameters (figure 12.2b).

The attention weights $\alpha[x_m, x_n]$ combine the values from different inputs. They are also sparse since there is only one weight for each ordered pair of inputs $(x_m, x_n)$, regardless of the size of these inputs (figure 12.2c). It follows that the number of attention weights has a quadratic dependence on the sequence length $N$, but is independent of the length $D$ of each input $x_n$.

### 12.2.2 Computing attention weights
In the previous section, we saw that the outputs result from two chained linear transformations; the value vectors $\beta_v + \Omega_vx_m$ are computed independently for each input $x_m$, and these vectors are combined linearly by the attention weights $\alpha[x_m, x_n]$. However, the overall self-attention computation is *nonlinear*. As we'll see shortly, the attention weights are themselves nonlinear functions of the input. This is an example of a *hypernetwork*, where one network branch computes the weights of another.

To compute the attention, we apply two more linear transformations to the inputs:

$$q_n = \beta_q + \Omega_qx_n$$
$$k_m = \beta_k + \Omega_kx_m,$$

where $\{q_n\}$ and $\{k_m\}$ are termed *queries* and *keys*, respectively. Then we compute dot products between the queries and keys and pass the results through a softmax function:

$$a[x_m, x_n] = \text{softmax}_{x_m} [k_m^T q_n]$$
$$= \frac{\exp[k_m^T q_n]}{\sum_{m'=1}^N \exp[k_{m'}^T q_n]},$$

so for each $x_n$, they are positive and sum to one (figure 12.3). For obvious reasons, this is known as *dot-product self-attention*.

The names "queries" and "keys" were inherited from the field of information retrieval and have the following interpretation: the dot product operation returns a measure of similarity between its inputs, so the weights $a[x_0, x_n]$ depend on the relative similarities between the $n^{th}$ query and all of the keys. The softmax function means that the key vectors "compete" with one another to contribute to the final result. The queries and keys must have the same dimensions. However, these can differ from the dimension of vectors "compete" with one another to contribute to the final result. The queries and keys must have the same dimensions. However, these can differ from the dimension of vectors "compete" with one another to contribute to the final result. The queries and keys must have the same dimensions. However, these can differ from the dimension of the values, which is usually the same size as the input, so the representation doesn’t change size.
