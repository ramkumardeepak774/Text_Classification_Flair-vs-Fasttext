# Text_Classification: flair vs Fastext

## Flair
Word embeddings: Flair uses pre-trained word embeddings, which are essentially vectors that represent the meaning of words in a lower-dimensional space. These word embeddings are learned using unsupervised methods, such as GloVe or FastText, and can be seen as a type of dimensionality reduction technique that captures the semantic meaning of words. Mathematically, each word w is represented as a vector x_w ∈ ℝ^d, where d is the dimensionality of the embedding space.

Document embeddings: To represent an entire document, Flair aggregates the word embeddings in some way, such as averaging them or passing them through a document-level LSTM network. In the case of the LSTM network, each word embedding is passed through the network, which has a series of hidden states h_t ∈ ℝ^h, where h is the number of hidden units in the LSTM. The final hidden state of the LSTM is then used as the document embedding, z ∈ ℝ^h.

LSTM network: The LSTM network is a type of RNN that allows the network to retain information from previous time steps and use it to influence the output at the current time step. The LSTM network is composed of several gates and cells that control the flow of information through the network. Mathematically, the LSTM network can be described using the following equations:

Input gate: i_t = σ(W_i [h_{t-1}, x_t] + b_i)
Forget gate: f_t = σ(W_f [h_{t-1}, x_t] + b_f)
Output gate: o_t = σ(W_o [h_{t-1}, x_t] + b_o)
Cell state: c_t = f_t * c_{t-1} + i_t * tanh(W_c [h_{t-1}, x_t] + b_c)
Hidden state: h_t = o_t * tanh(c_t)
Here, σ is the sigmoid activation function, tanh is the hyperbolic tangent activation function, and * denotes element-wise multiplication. The LSTM network is trained using backpropagation through time (BPTT), which is a variant of backpropagation that can handle sequences of variable length.

Softmax function: Once the document embedding z has been computed using the LSTM network, it is passed through a linear layer followed by a softmax function to obtain a probability distribution over the classes. Mathematically, the softmax function can be described as:

y_i = exp(W_y z + b_y)_i / ∑_j exp(W_y z + b_y)_j
Here, W_y and b_y are the weights and bias of the linear layer, and y_i is the predicted probability of the document belonging to class i.

Training: To train the text classification model, Flair uses a labeled dataset of documents and their corresponding class labels. The objective is to minimize the cross-entropy loss between the predicted class probabilities and the true class labels. Mathematically, the cross-entropy loss can be described as:

L(y, t) = -∑_i t_i log y_i
Here, t is the one-hot encoded true class label, and y is the predicted probability distribution over the classes. The text classification model is trained using stochastic gradient descent (SGD) or one of its variants, such as Adam, to minimize the cross-entropy loss.

Inference: To predict the class of a new, unseen document, Flair passes the document through the trained LSTM network and computes the class probabilities using the softmax function. The predicted




## Fasttext 

Bag-of-words representation: FastText represents a document as a bag-of-words, which is a simple way of capturing the word frequencies in the document. Mathematically, given a vocabulary of size V, a document d is represented as a vector x_d ∈ ℝ^V, where each entry x_d[i] corresponds to the frequency of the i-th word in the vocabulary in the document.

Word embeddings: FastText uses pre-trained word embeddings, which are essentially vectors that represent the meaning of words in a lower-dimensional space. These word embeddings are learned using unsupervised methods, such as Skip-gram or CBOW, and can be seen as a type of dimensionality reduction technique that captures the semantic meaning of words. Mathematically, each word w is represented as a vector x_w ∈ ℝ^d, where d is the dimensionality of the embedding space.

Hierarchical softmax: To predict the class of a document, FastText uses a hierarchical softmax classifier, which is a variant of the standard softmax classifier that reduces the computational complexity. The hierarchical softmax classifier uses a binary tree structure to represent the classes, where each leaf node corresponds to a class and each internal node corresponds to a binary decision that determines whether to take the left or right subtree. Mathematically, given a document vector x_d and a class c represented by a leaf node in the binary tree, the probability of the document belonging to the class is given by:

P(c | x_d) = ∏_{j∈path(c)} σ(x_d^T v_j)
Here, σ is the sigmoid activation function, v_j is the vector associated with node j in the binary tree, and path(c) is the set of nodes on the path from the root node to the leaf node representing class c. The hierarchical softmax classifier is trained using stochastic gradient descent (SGD) or one of its variants, such as Adam, to minimize the negative log-likelihood of the training data.

N-gram features: In addition to the bag-of-words representation, FastText also includes n-gram features to capture the local context of the words in the document. An n-gram is a contiguous sequence of n words in the document, and n-gram features capture the frequency of each n-gram in the document. Mathematically, given a vocabulary of size V and a maximum n-gram length N, a document d is represented as a vector x_d ∈ ℝ^(V + V_N), where V_N is the number of distinct n-grams in the vocabulary. Each entry x_d[i] corresponds to the frequency of the i-th word or n-gram in the document.

Subword embeddings: To handle out-of-vocabulary words and rare words, FastText also includes subword embeddings, which are essentially vectors that represent the meaning of character n-grams in a lower-dimensional space. Subword embeddings can be seen as a type of character-level word embedding that captures the morphological structure of words. Mathematically, each character n-gram s is represented as a vector x_s ∈ ℝ^d', where d' is the dimensionality of the subword embedding space.

Overall, FastText combines bag-of-words, n-gram, and subword features with pre-trained word and subword embeddings to obtain a powerful text classification model. The model is trained using a hierarchical softmax classifier with SGD or one of its variants, and can handle out-of-vocabulary words and rare words using the subword embeddings.





