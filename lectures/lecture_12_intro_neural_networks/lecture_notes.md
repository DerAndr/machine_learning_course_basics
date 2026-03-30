# Lecture 12 Notes: Introduction to Neural Networks and Deep Learning

> Lecture number: 12
> Lecture slug: `lecture_12_intro_neural_networks`
> Role: student-facing recap and revision notes
> Use this file: after the lecture, before or alongside the practice notebook
> Related files: `README.md`, `slides/lecture.pdf`, `assignment/practice.ipynb`
> Source basis: lecture slides and practical notebooks
> Last updated: 2026-03-31

## 1. What This Lecture Is About

This lecture introduces the core ideas behind neural networks and deep learning:

- what a neural network is,
- why deep learning matters,
- how neurons, layers, and activation functions work,
- why single-layer perceptrons are limited,
- how backpropagation trains networks,
- what modern extensions such as CNNs, RNNs, and Transformers are trying to solve.

The main goal is not to make students experts in deep learning after one lecture. The goal is to give a solid conceptual and mathematical foundation so later architectures make sense instead of looking like disconnected tricks.

## 2. Why Deep Learning Matters

Deep learning is a subset of machine learning that uses multiple layers to learn hierarchical representations from data.

The keyword here is hierarchical.

Instead of hand-crafting all features manually, deep models can learn layered representations:

- early layers capture simple patterns,
- deeper layers combine them into more abstract concepts.

In image tasks, this might look like:

- edges,
- corners,
- textures,
- parts of objects,
- full object identity.

In language tasks, this might look like:

- tokens,
- local context,
- syntax,
- semantic patterns,
- long-range dependencies.

The lecture emphasizes that deep learning matters because it dominates many major benchmarks in:

- image recognition,
- speech recognition,
- natural language processing,
- recommendation,
- generative modeling.

It is also a strongly interdisciplinary area. The slides correctly tie it to:

- linear algebra,
- calculus,
- probability,
- numerical analysis,
- high-performance computing on GPUs or TPUs.

Students should see deep learning not as “magic AI,” but as a combination of mathematical structure, large datasets, and scalable optimization.

## 3. Historical Overview

The historical part of the lecture gives important context.

### Early ideas

The idea of an artificial neuron goes back to McCulloch and Pitts in the 1940s. The perceptron, introduced by Frank Rosenblatt in the 1950s, was the first major learning-based neural model.

The perceptron was important because it showed that a system could learn decision boundaries from data. But it also had a key limitation: it was essentially a linear separator.

### The XOR problem

The XOR problem became the symbolic demonstration of that limitation.

For XOR:

- output is 1 when the inputs differ,
- output is 0 when the inputs are the same.

The four points cannot be separated by one straight line in input space. That means a single-layer perceptron cannot solve XOR.

This mattered historically because it showed that shallow linear systems had clear expressive limits.

### Backpropagation revival

The field regained momentum when backpropagation became practically usable for training multilayer networks. That made hidden-layer training feasible and reopened the path toward nonlinear representation learning.

### Sequence modeling and LSTM

The lecture also highlights Jürgen Schmidhuber and the LSTM architecture with Hochreiter. This matters because recurrent models exposed a central training issue in deep learning:

- long-range dependencies are hard to learn,
- gradients may vanish or explode,
- gating mechanisms can make sequential learning more stable.

### Modern deep learning era

The more recent wave includes:

- CNNs for computer vision,
- RNNs, LSTMs, and GRUs for sequence data,
- Transformers for large-scale sequence modeling and attention-based reasoning.

Students should take away that deep learning did not emerge all at once. It progressed through repeated cycles of:

- theoretical ideas,
- optimization advances,
- hardware improvements,
- dataset growth.

## 4. Mathematical Foundations

The lecture explicitly frames deep learning around three mathematical pillars.

### 1. Linear algebra

Neural networks are built from vector and matrix operations.

For a layer with input vector \(x\), weight matrix \(W\), and bias \(b\), the pre-activation is:

\[
z = Wx + b
\]

This is one of the most important equations in the whole topic.

Why?

Because nearly every feedforward layer in deep learning is some variant of:

- linear transformation,
- then nonlinearity.

When data is processed in batches, matrix operations become especially important because modern hardware is optimized for them.

### 2. Calculus

Training requires computing how the loss changes when a parameter changes. That is a derivative question.

The network is a composition of many functions. To update early weights, we must propagate error information backward through all intermediate computations.

This is exactly what the chain rule enables.

### 3. Optimization

Once gradients are known, we need an algorithm that updates parameters to reduce loss.

The lecture mentions:

- gradient descent,
- stochastic gradient descent,
- related optimization variants,
- convergence considerations.

The optimization perspective is crucial: a neural network is not only a function class, but also a trainable system whose success depends on whether optimization works well enough in practice.

## 5. Structure of a Neural Network

The lecture breaks this down into neurons, layers, and connections.

### Neurons

A neuron computes a weighted combination of inputs, adds a bias, and applies an activation function.

The core form is:

\[
z_j = \sum_{i=1}^{n} w_{ij}x_i + b_j
\]

and the neuron output is:

\[
o_j = \phi(z_j)
\]

where:

- \(w_{ij}\) are weights,
- \(x_i\) are inputs,
- \(b_j\) is bias,
- \(\phi\) is the activation function.

Bias is important because without it, the neuron’s decision surface is forced through the origin. Bias shifts the activation and increases flexibility.

### Layers

The lecture distinguishes:

- input layer,
- hidden layers,
- output layer.

#### Input layer

Receives raw features. It usually does not perform computation beyond passing values forward.

#### Hidden layers

Transform the representation. This is where feature learning happens.

#### Output layer

Depends on the task:

- one unit for scalar regression,
- one sigmoid-style output for binary classification,
- multiple softmax outputs for multiclass classification.

### Connections

In a fully connected network, each neuron in one layer connects to every neuron in the next layer.

This is expressive, but it also increases parameter count. That becomes important later when comparing MLPs with architectures such as CNNs, which use local connectivity and parameter sharing.

## 6. Activation Functions

Activation functions are essential because they introduce nonlinearity.

Without nonlinearity, stacking layers would collapse into a single linear transformation.

That is one of the most important ideas in the lecture.

If every layer were purely linear:

\[
W_3(W_2(W_1x)) = W'x
\]

for some equivalent matrix \(W'\). No matter how many linear layers you stack, the result is still linear.

### Linear activation

\[
\phi(x) = x
\]

Useful mainly in regression output layers when unrestricted real-valued predictions are needed.

Not useful in hidden layers if the goal is complex nonlinear modeling.

### Sigmoid

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

Properties:

- output range \((0, 1)\),
- interpretable as a probability-like output,
- commonly used in binary classification output layers.

Main limitation:

- vanishing gradients when activations saturate near 0 or 1.

Its derivative is

\[
\sigma'(x) = \sigma(x)(1-\sigma(x))
\]

which is always at most \(0.25\). In deep compositions, repeatedly multiplying by small local derivatives can make gradients shrink rapidly.

### Tanh

\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

Properties:

- output range \((-1, 1)\),
- zero-centered,
- often preferable to sigmoid in hidden layers historically.

Main limitation:

- still suffers from vanishing gradients for large positive or negative inputs.

Its derivative is

\[
\tanh'(x) = 1 - \tanh^2(x)
\]

which is larger near zero than the sigmoid derivative, but still approaches zero in saturated regions.

### ReLU

\[
\text{ReLU}(x) = \max(0, x)
\]

Properties:

- simple,
- computationally efficient,
- sparse activation pattern,
- dominant choice for many hidden layers.

Main limitation:

- dead neurons, where a unit outputs zero for many inputs and stops updating effectively.

Its derivative is simple:

\[
\frac{d}{dx}\mathrm{ReLU}(x) =
\begin{cases}
1 & x > 0 \\
0 & x < 0
\end{cases}
\]

This is one reason ReLU-like activations often optimize more easily than sigmoid or tanh in deep hidden stacks.

### Leaky ReLU

\[
\phi(x) =
\begin{cases}
x & x > 0 \\
\alpha x & x \le 0
\end{cases}
\]

This addresses the dead-neuron problem by allowing a small negative slope.

### Softmax

For class \(i\):

\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Properties:

- converts logits into a probability distribution,
- outputs sum to 1,
- standard choice for multiclass classification output layers.

Students should remember:

- sigmoid is usually for binary output,
- softmax is usually for multiclass output,
- ReLU-like functions are usually for hidden layers.

## 7. Why MLPs Matter

The Multi-Layer Perceptron, or MLP, is the foundational feedforward neural network.

It consists of:

- an input layer,
- one or more hidden layers,
- an output layer,
- nonlinear activations between layers.

The lecture emphasizes that adding hidden layers allows the model to solve problems that a single-layer perceptron cannot.

This is the conceptual jump from shallow linear classification to nonlinear representation learning.

### Why the hidden layer changes everything

The hidden layer creates intermediate transformations of the input space. These transformations can make a nonlinearly separable problem become separable in a learned feature space.

That is the intuition behind why XOR becomes solvable.

## 8. XOR as a Turning Point

This is one of the most important examples in the lecture.

The XOR truth table is:

- \((0, 0) \to 0\)
- \((0, 1) \to 1\)
- \((1, 0) \to 1\)
- \((1, 1) \to 0\)

A single linear separator cannot split the positive and negative examples correctly.

But once a hidden layer is added, the network can learn intermediate nonlinear transformations that make the problem solvable.

The XOR example matters for two reasons:

- historically, it exposed the weakness of single-layer perceptrons,
- conceptually, it shows why hidden layers and nonlinear activation are necessary.

## 9. Universal Approximation Theorem

The lecture includes the universal approximation theorem, which is important but often misunderstood.

### What it says

A feedforward neural network with:

- one hidden layer,
- enough neurons,
- a suitable non-linear activation,

can approximate any continuous function on a compact domain to arbitrary accuracy.

### What it does not say

It does not say:

- that one hidden layer is always practical,
- that training will be easy,
- that the network will generalize well,
- that the required width will be reasonable,
- that deep architectures are unnecessary.

This is a theorem about expressive capacity, not about optimization quality or sample efficiency.

That distinction is extremely important.

In practice, deeper architectures are often preferred because they can represent useful structure more efficiently than a very wide shallow network.

## 10. Forward Propagation

Forward propagation is the process of sending inputs through the network to compute predictions.

For each layer:

1. compute weighted sums,
2. add biases,
3. apply activation,
4. pass output to the next layer.

This continues until the final prediction is produced.

The practical notebook illustrates this idea with simple layer computations and activation-curve visualizations.

Students should think of forward propagation as “evaluation of the computational graph.”

## 11. Loss Functions

Training needs a scalar objective to optimize.

The lecture highlights common losses:

### Regression losses

#### Mean Squared Error

\[
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

Strongly penalizes large errors.

#### Mean Absolute Error

\[
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
\]

More robust to outliers than MSE.

### Classification losses

#### Cross-entropy

Widely used because it compares predicted class probabilities with the true label distribution and provides strong gradients for probabilistic classification training.

The notebook that visualizes MSE, MAE, and cross-entropy is especially useful here because it helps students see that different losses shape the optimization landscape differently.

## 12. Backpropagation

Backpropagation is the algorithm used to compute gradients efficiently in a neural network.

This is one of the core ideas of the entire lecture.

### The basic training loop

1. perform a forward pass,
2. compute the loss,
3. compute gradients of the loss with respect to parameters,
4. update weights,
5. repeat.

### Why backpropagation matters

A deep network may have millions of parameters. We need a systematic way to compute:

\[
\frac{\partial L}{\partial w}
\]

for every parameter \(w\).

Doing this separately from scratch for each weight would be computationally impossible. Backpropagation reuses intermediate derivatives efficiently.

## 13. The Chain Rule

The chain rule is the mathematical foundation of backpropagation.

For a composition of functions:

\[
f(g(x))
\]

the derivative is:

\[
\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)
\]

In deep learning, the network output is a composition of many such functions:

- affine transforms,
- activations,
- losses,
- maybe normalization and other operations.

The chain rule tells us how a small change in an early parameter affects the final loss through all downstream computations.

The lecture’s computational graph examples are valuable because they turn an abstract calculus rule into a concrete flow of values and gradients.

## 14. Weight Updates and Gradient Descent

Once gradients are computed, parameters are updated.

A basic gradient descent step is:

\[
w \leftarrow w - \eta \frac{\partial L}{\partial w}
\]

where:

- \(w\) is a weight,
- \(L\) is the loss,
- \(\eta\) is the learning rate.

### Why the learning rate matters

If the learning rate is too small:

- training becomes slow,
- optimization may stall.

If it is too large:

- updates may overshoot,
- training may oscillate or diverge.

The slides mention variants such as SGD and other optimizers. Students should understand that optimizer choice is not secondary. It strongly affects training behavior.

It is also useful to distinguish:

- batch gradient descent: one update using the full dataset,
- stochastic gradient descent: one update per example,
- mini-batch gradient descent: one update per small batch.

Modern deep learning is usually trained with mini-batches because they balance gradient quality, memory efficiency, and hardware throughput.

## 15. Vanishing and Exploding Gradients

This is one of the main optimization challenges introduced in the lecture.

### Vanishing gradients

Gradients become extremely small as they are propagated backward through many layers.

Consequences:

- early layers learn very slowly,
- long-range dependencies become hard to capture,
- sigmoid and tanh can worsen the issue because of saturation.

Mathematically, backpropagation multiplies local derivatives through many layers. If those derivatives are frequently less than 1 in magnitude, the product can decay exponentially with depth.

### Exploding gradients

Gradients become excessively large.

Consequences:

- unstable updates,
- numerical problems,
- diverging loss.

The opposite failure happens when repeated Jacobian factors become too large, causing gradient norms to blow up instead of decay.

These issues were major obstacles in early deep learning and remain important in recurrent settings.

Students should connect this back to:

- why ReLU became so popular,
- why LSTMs and GRUs were important,
- why initialization and optimizer design matter.

## 16. Practical Notebook Map

This lecture’s notebooks are quite useful for building intuition.

### 1. `DL Intro.ipynb`

This notebook focuses on conceptual visualizations and small mathematical examples:

- plotting activation functions,
- showing where nonlinearity comes from,
- illustrating the chain rule with a tiny forward/backward computation,
- drawing a computational graph,
- comparing loss behaviors.

This is a good notebook for intuition. It helps students see that deep learning is built from repeated simple operations, not mysterious black-box magic.

### 2. `NNs.ipynb`

This notebook is the more applied practical piece. It includes:

- classical dataset setup with the Wine dataset,
- preprocessing with scaling,
- an MNIST section,
- a PyTorch example,
- a first taste of neural-network training in a modern framework.

This is useful because it bridges the lecture from abstract theory to real tooling. Students can see that:

- preprocessing still matters,
- neural networks are trained like other ML systems in a workflow,
- PyTorch introduces a practical model-building interface around tensors, modules, losses, and optimizers.

## 17. Modern Extensions

The lecture ends by situating MLPs within the broader neural network landscape.

### CNNs

Convolutional Neural Networks are designed for grid-like data such as images.

Key ideas:

- local receptive fields,
- parameter sharing,
- translation-aware pattern extraction.

Why they matter:

- far more efficient than fully connected networks on images,
- preserve spatial structure,
- historically central to modern computer vision.

### RNNs, LSTMs, and GRUs

These are designed for sequence data:

- text,
- speech,
- time series.

RNNs process data sequentially and maintain a hidden state. LSTMs and GRUs introduce gating to better preserve information across long contexts.

This links back directly to the vanishing gradient issue discussed earlier.

### Transformers

Transformers replaced recurrence with attention-based mechanisms for many tasks.

Why that mattered:

- long-range dependencies became easier to model,
- parallelization improved,
- large-scale language modeling became much more effective.

The lecture rightly frames Transformers as a major turning point in modern AI.

## 18. Important Conceptual Distinctions Students Should Remember

### Deep learning vs. ordinary machine learning

Deep learning is still machine learning, but it relies more heavily on learned representations from multilayer neural networks.

### Representation power vs. trainability

A network may be expressive enough in theory, but still difficult to train in practice.

### Approximation vs. generalization

Approximating a function well on training data does not guarantee generalization to unseen data.

### More layers vs. better model

Depth increases modeling power, but it also introduces:

- optimization challenges,
- higher compute cost,
- overfitting risk,
- more tuning complexity.

## 19. What Students Should Be Able to Explain After This Lecture

### What is a neuron?

A unit that computes a weighted sum of inputs plus bias and then applies an activation function.

### Why are activation functions necessary?

Because without nonlinearity, a multilayer network collapses into one linear transformation.

### Why can a perceptron not solve XOR?

Because XOR is not linearly separable.

### What does the universal approximation theorem actually guarantee?

That a sufficiently wide one-hidden-layer network can approximate continuous functions, but not that training is easy or efficient.

### What does backpropagation do?

It computes gradients of the loss with respect to network parameters efficiently using the chain rule.

### Why do vanishing gradients matter?

Because early layers stop receiving useful learning signals, making deep or sequential training difficult.

### Why did architectures like CNNs, LSTMs, and Transformers appear?

Because fully connected MLPs are not always the most efficient or stable way to model images, sequences, and long-range dependencies.

## 20. Key Takeaways

- Neural networks are compositions of linear transformations and nonlinear activations.
- Hidden layers make nonlinear function learning possible.
- The XOR problem explains why single-layer perceptrons are limited.
- Backpropagation plus gradient-based optimization is the engine of training.
- Deep learning success depends not only on theory, but also on optimization and computation.
- CNNs, recurrent models, and Transformers are specialized responses to different data structures and learning challenges.

## 21. Quick Revision Questions

1. Why does stacking linear layers without nonlinear activation not increase expressive power?
2. Why is XOR such an important example in the history of neural networks?
3. What exactly does the chain rule contribute to backpropagation?
4. Why can sigmoid and tanh make optimization harder in deep networks?
5. Why are CNNs usually more suitable than plain MLPs for image data?
