# LSTM: Mathematical Derivation

This note derives the **modern LSTM architecture with a forget gate**, from its forward equations through backpropagation through time (BPTT) and parameter updates. The derivation uses column vectors and elementwise gate operations.

## 1. Notation and dimensions

Let:

- \(D\): input feature dimension
- \(H\): hidden-state and cell-state dimension
- \(K\): number of output classes
- \(T\): sequence length

At time step \(t\):

\[
\mathbf{x}_t\in\mathbb{R}^{D\times 1},\qquad
\mathbf{h}_t,\mathbf{c}_t\in\mathbb{R}^{H\times 1}
\]

The hidden state and cell state have the same dimension because the hidden-state equation applies an elementwise operation between them.

For each gate or candidate transformation \(g\in\{i,f,o,c\}\):

\[
\mathbf{W}_g\in\mathbb{R}^{H\times D},\quad
\mathbf{U}_g\in\mathbb{R}^{H\times H},\quad
\mathbf{b}_g\in\mathbb{R}^{H\times 1}
\]

Here, \(c\) denotes the candidate-memory transformation, not the cell state itself.

Notation:

- \(\sigma(\cdot)\): sigmoid, applied elementwise
- \(\tanh(\cdot)\): hyperbolic tangent, applied elementwise
- \(\odot\): elementwise multiplication
- \(\mathbf{1}\): all-ones vector of the required dimension

The initial states are commonly set to zero:

\[
\mathbf{h}_0=\mathbf{0},\qquad \mathbf{c}_0=\mathbf{0}
\]

## 2. Forward propagation

At each time step, the LSTM computes four transformations.

### 2.1 Input gate

The input gate controls how much candidate information is admitted into the cell state.

\[
\mathbf{a}_{i,t}
=\mathbf{W}_i\mathbf{x}_t+\mathbf{U}_i\mathbf{h}_{t-1}+\mathbf{b}_i
\]

\[
\boxed{\mathbf{i}_t=\sigma(\mathbf{a}_{i,t})}
\]

### 2.2 Forget gate

The forget gate controls how much of the previous cell state is retained.

\[
\mathbf{a}_{f,t}
=\mathbf{W}_f\mathbf{x}_t+\mathbf{U}_f\mathbf{h}_{t-1}+\mathbf{b}_f
\]

\[
\boxed{\mathbf{f}_t=\sigma(\mathbf{a}_{f,t})}
\]

### 2.3 Candidate memory

The candidate proposes new content for the cell state.

\[
\mathbf{a}_{c,t}
=\mathbf{W}_c\mathbf{x}_t+\mathbf{U}_c\mathbf{h}_{t-1}+\mathbf{b}_c
\]

\[
\boxed{\tilde{\mathbf{c}}_t=\tanh(\mathbf{a}_{c,t})}
\]

### 2.4 Cell-state update

The new cell state combines retained previous memory and admitted candidate information:

\[
\boxed{
\mathbf{c}_t
=\mathbf{f}_t\odot\mathbf{c}_{t-1}
+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t
}
\]

The cell state is not constrained to the interval \([-1,1]\); it can accumulate information over time.

### 2.5 Output gate and hidden state

The output gate controls how much of the transformed cell state is exposed as the hidden state.

\[
\mathbf{a}_{o,t}
=\mathbf{W}_o\mathbf{x}_t+\mathbf{U}_o\mathbf{h}_{t-1}+\mathbf{b}_o
\]

\[
\boxed{\mathbf{o}_t=\sigma(\mathbf{a}_{o,t})}
\]

\[
\boxed{\mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)}
\]

Both \(\mathbf{c}_t\) and \(\mathbf{h}_t\) are carried to the next time step, but they have different roles: the cell state is the internal memory, while the hidden state is the exposed representation.

## 3. Sequence-level output and loss

For sequence classification, use the final hidden state \(\mathbf{h}_T\).

The output layer produces logits:

\[
\mathbf{z}=\mathbf{W}_y\mathbf{h}_T+\mathbf{b}_y
\]

with:

\[
\mathbf{W}_y\in\mathbb{R}^{K\times H},\qquad
\mathbf{b}_y,\mathbf{z}\in\mathbb{R}^{K\times1}
\]

Softmax converts logits into class probabilities:

\[
p_k=\frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}
\]

Let \(\mathbf{y}\in\mathbb{R}^{K\times1}\) be the target distribution, with \(\sum_k y_k=1\). Categorical cross-entropy is:

\[
\boxed{
L=-\sum_{k=1}^{K}y_k\log(p_k)
=-\mathbf{y}^{T}\log(\mathbf{p})
}
\]

The logarithm is applied elementwise in the vector expression. For ordinary single-label classification, \(\mathbf{y}\) is usually one-hot.

## 4. Backpropagation through the output layer

Define the logit gradient:

\[
\boxed{
\boldsymbol{\delta}_z
=\frac{\partial L}{\partial\mathbf{z}}
=\mathbf{p}-\mathbf{y}
}
\]

The output-layer parameter gradients are:

\[
\boxed{
\frac{\partial L}{\partial\mathbf{W}_y}
=\boldsymbol{\delta}_z\mathbf{h}_T^T
}
\]

\[
\boxed{
\frac{\partial L}{\partial\mathbf{b}_y}
=\boldsymbol{\delta}_z
}
\]

The gradient arriving at the final hidden state is:

\[
\boxed{
\bar{\mathbf{h}}_T
=\frac{\partial L}{\partial\mathbf{h}_T}
=\mathbf{W}_y^T\boldsymbol{\delta}_z
}
\]

A bar denotes a gradient with respect to the variable underneath it.

## 5. Backpropagation through one LSTM cell

At a given time step, the incoming gradients are propagated through the hidden-state equation and the cell-state update.

### 5.1 Hidden-state contributions

From:

\[
\mathbf{h}_t=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)
\]

the gradient with respect to the activated output gate is:

\[
\boxed{
\bar{\mathbf{o}}_t
=\bar{\mathbf{h}}_t\odot\tanh(\mathbf{c}_t)
}
\]

The hidden-state path also contributes to the cell-state gradient:

\[
\boxed{
\bar{\mathbf{c}}_t
=
\bar{\mathbf{c}}_t^{\,\mathrm{future}}
+
\bar{\mathbf{h}}_t\odot\mathbf{o}_t
\odot\left(\mathbf{1}-\tanh^2(\mathbf{c}_t)\right)
}
\]

Here, \(\bar{\mathbf{c}}_t^{\,\mathrm{future}}\) is the direct gradient arriving through the cell-state connection from the next time step. At the final time step for sequence-level classification, this direct future contribution is zero.

### 5.2 Contributions through the cell-state update

Recall:

\[
\mathbf{c}_t
=\mathbf{f}_t\odot\mathbf{c}_{t-1}
+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t
\]

The gradients with respect to the activated gate and candidate values are:

\[
\boxed{\bar{\mathbf{f}}_t
=\bar{\mathbf{c}}_t\odot\mathbf{c}_{t-1}}
\]

\[
\boxed{\bar{\mathbf{i}}_t
=\bar{\mathbf{c}}_t\odot\tilde{\mathbf{c}}_t}
\]

\[
\boxed{\bar{\tilde{\mathbf{c}}}_t
=\bar{\mathbf{c}}_t\odot\mathbf{i}_t}
\]

The direct cell-state contribution passed to the previous time step is:

\[
\boxed{
\bar{\mathbf{c}}_{t-1}^{\,\mathrm{future}}
=\bar{\mathbf{c}}_t\odot\mathbf{f}_t
}
\]

### 5.3 Gradients through the gate activations

Using:

\[
\sigma'(a)=\sigma(a)(1-\sigma(a))
\]

the sigmoid preactivation gradients are:

\[
\boxed{
\boldsymbol{\delta}_{i,t}
=\bar{\mathbf{i}}_t\odot\mathbf{i}_t
\odot(\mathbf{1}-\mathbf{i}_t)
}
\]

\[
\boxed{
\boldsymbol{\delta}_{f,t}
=\bar{\mathbf{f}}_t\odot\mathbf{f}_t
\odot(\mathbf{1}-\mathbf{f}_t)
}
\]

\[
\boxed{
\boldsymbol{\delta}_{o,t}
=\bar{\mathbf{o}}_t\odot\mathbf{o}_t
\odot(\mathbf{1}-\mathbf{o}_t)
}
\]

Using:

\[
\tanh'(a)=1-\tanh^2(a)
\]

the candidate preactivation gradient is:

\[
\boxed{
\boldsymbol{\delta}_{c,t}
=\bar{\tilde{\mathbf{c}}}_t
\odot(\mathbf{1}-\tilde{\mathbf{c}}_t^2)
}
\]

Each \(\boldsymbol{\delta}_{g,t}\) is the gradient with respect to the corresponding affine preactivation.

## 6. Parameter gradients at one time step

For each \(g\in\{i,f,o,c\}\), the affine transformation is:

\[
\mathbf{a}_{g,t}
=\mathbf{W}_g\mathbf{x}_t
+\mathbf{U}_g\mathbf{h}_{t-1}
+\mathbf{b}_g
\]

Therefore, the parameter-gradient contributions at time \(t\) are:

\[
\boxed{
\frac{\partial L}{\partial\mathbf{W}_g}\bigg|_t
=\boldsymbol{\delta}_{g,t}\mathbf{x}_t^T
}
\]

\[
\boxed{
\frac{\partial L}{\partial\mathbf{U}_g}\bigg|_t
=\boldsymbol{\delta}_{g,t}\mathbf{h}_{t-1}^T
}
\]

\[
\boxed{
\frac{\partial L}{\partial\mathbf{b}_g}\bigg|_t
=\boldsymbol{\delta}_{g,t}
}
\]

## 7. Backpropagation through time

The same recurrent parameters are reused at every time step. Consequently, gradients from all time steps contribute to the same parameter gradients.

The gradient arriving at the previous hidden state through the four recurrent transformations is:

\[
\boxed{
\bar{\mathbf{h}}_{t-1}
=
\mathbf{U}_i^T\boldsymbol{\delta}_{i,t}
+\mathbf{U}_f^T\boldsymbol{\delta}_{f,t}
+\mathbf{U}_o^T\boldsymbol{\delta}_{o,t}
+\mathbf{U}_c^T\boldsymbol{\delta}_{c,t}
}
\]

The cell-state gradient is carried through its direct recurrent connection:

\[
\boxed{
\bar{\mathbf{c}}_{t-1}^{\,\mathrm{future}}
=\bar{\mathbf{c}}_t\odot\mathbf{f}_t
}
\]

At the next iteration, the hidden-state contribution at time \(t-1\) is added to this direct cell-state contribution using the hidden-state equation in Section 5.1.

Starting with:

\[
\bar{\mathbf{h}}_T=\mathbf{W}_y^T(\mathbf{p}-\mathbf{y}),
\qquad
\bar{\mathbf{c}}_T^{\,\mathrm{future}}=\mathbf{0}
\]

these calculations are repeated in reverse order:

\[
T,T-1,\ldots,1
\]

This is backpropagation through time (BPTT).

## 8. Accumulating shared parameter gradients

Since the same gate/candidate parameters are used at every time step, their total gradients are sums of the per-step contributions.

For every \(g\in\{i,f,o,c\}\):

\[
\boxed{
\frac{\partial L}{\partial\mathbf{W}_g}
=\sum_{t=1}^{T}\boldsymbol{\delta}_{g,t}\mathbf{x}_t^T
}
\]

\[
\boxed{
\frac{\partial L}{\partial\mathbf{U}_g}
=\sum_{t=1}^{T}\boldsymbol{\delta}_{g,t}\mathbf{h}_{t-1}^T
}
\]

\[
\boxed{
\frac{\partial L}{\partial\mathbf{b}_g}
=\sum_{t=1}^{T}\boldsymbol{\delta}_{g,t}
}
\]

For a batch of \(N\) sequences, calculate the corresponding per-sequence gradients and average them when the batch loss is defined as a mean.

## 9. Parameter updates

The trainable parameter tensors are:

- LSTM transformations: \(\mathbf{W}_i,\mathbf{U}_i,\mathbf{b}_i\); \(\mathbf{W}_f,\mathbf{U}_f,\mathbf{b}_f\); \(\mathbf{W}_o,\mathbf{U}_o,\mathbf{b}_o\); \(\mathbf{W}_c,\mathbf{U}_c,\mathbf{b}_c\).
- Output layer: \(\mathbf{W}_y,\mathbf{b}_y\).

There are 14 trainable parameter tensors in this architecture, excluding an optional embedding layer.

For any trainable parameter tensor \(\theta\), gradient descent updates it as:

\[
\boxed{
\theta_{\mathrm{new}}
=\theta_{\mathrm{old}}-\eta\nabla_\theta L
}
\]

where \(\eta\) is the learning rate.

The hidden states, cell states, gates and candidate values are activations computed during the forward pass; they are not directly updated as model parameters.

## 10. Forward and backward flow summary

The sequence-classification forward path is:

\[
\mathbf{x}_{1:T}
\longrightarrow
\text{LSTM}
\longrightarrow
\mathbf{h}_T
\longrightarrow
\mathbf{z}
\longrightarrow
\mathbf{p}
\longrightarrow
L
\]

The backward path starts at the loss, calculates output-layer gradients, propagates through the final hidden state and LSTM cell, and then moves backwards through the sequence using BPTT. Parameter contributions are accumulated across time before the trainable tensors are updated.

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
2. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. *Neural Computation*, 12(10), 2451–2471. https://doi.org/10.1162/089976600300015015

This derivation focuses on the modern LSTM with a forget gate. The original 1997 LSTM architecture did not include the forget gate; it was introduced in later work.
