Chapter 8 described how to measure model performance and identified that there could be a significant performance gap between the training and test data. Possible reasons for this discrepancy include: (i) the model describes statistical peculiarities of the training data that are not representative of the true mapping from input to output (overfitting), and (ii) the model is unconstrained in areas with no training examples, leading to sub- optimal predictions.

This chapter discusses regularization techniques. These are a family of methods that reduce the generalization gap between training and test performance. Strictly speaking, regularization involves adding explicit terms to the loss function that favor certain pa- rameter choices. However, in machine learning, this term is commonly used to refer to any strategy that improves generalization.

We start by considering regularization in its strictest sense. Then we show how the stochastic gradient descent algorithm itself favors certain solutions. This is known as implicit regularization. Following this, we consider a set of heuristic methods that improve test performance. These include early stopping, ensembling, dropout, label smoothing, and transfer learning.

## 9.1 Explicit regularization

Consider fitting a model $f(x, \phi)$ with parameters $\phi$ using a training set $\{x_i, y_i\}$ of input/output pairs. We seek the minimum of the loss function $L[\phi]$ :

$$
\hat{\phi} = argmin_\phi [L[\phi]]
$$

$$
= argmin_\phi \left[ \sum_{i=1}^I l(x_i, y_i) \right],
$$

where the individual terms $l(x_i, y_i)$ measure the mismatch between the network predictions $f(x_i, \phi)$ and output targets $y_i$ for each training pair. To bias this minimization toward certain solutions, we include an additional term:

$$
\hat{\phi} = argmin_\phi \left[ \sum_{i=1}^I l(x_i, y_i) + \lambda \cdot g(\phi) \right],
$$

where $g(\phi)$ is a function that returns a scalar that takes a larger value when the parameters are less preferred. The term $\lambda$ is a positive scalar that controls the relative contribution of the original loss function and the regularization term. The minima of the regularized loss function usually differ from those in the original, so the training procedure converges to different parameter values (figure 9.1).

## 9.1.1 Probabilistic interpretation

Regularization can be viewed from a probabilistic perspective. Section 5.1 shows how loss functions are constructed from the maximum likelihood criterion:

$$
\hat{\phi} = argmax_\phi \left[ \prod_{i=1}^I Pr(y_i|x_i, \phi) \right].
$$

The regularization term can be considered as a prior $Pr(\phi)$ that represents knowledge about the parameters before we observe the data and we now have the maximum a posteriori or MAP criterion:

$$
\hat{\phi} = \argmax_\phi \left[ \prod_{i=1}^I Pr(y_i|x_i, \phi) Pr(\phi) \right].
$$

Moving back to the negative log-likelihood loss function by taking the log and multiplying by minus one, we see that $\lambda \cdot g(\phi) = -\log(Pr(\phi))$.

## 9.1.2 L2 regularization

This discussion has sidestepped the question of *which* solutions the regularization term should penalize (or equivalently that the prior should favor). Since neural networks are used in an extremely broad range of applications, these can only be very generic preferences. The most commonly used regularization term is the L2 norm, which penalizes the sum of the squares of the parameter values:

$$
\hat{\phi} = \argmin_\phi \left[ \sum_{i=1}^I l(x_i, y_i) + \lambda \sum_j \phi_j^2 \right],
$$
where j indexes the parameters. This is also referred to as Tikhonov regularization or ridge regression, or (when applied to matrices) Frobenius norm regularization.

For neural networks, L2 regularization is usually applied to the weights but not the biases and is hence referred to as a weight decay term. The effect is to encourage smaller weights, so the output function is smoother. To see this, consider that the output prediction is a weighted sum of the activations at the last hidden layer. If the weights have a smaller magnitude, the output will vary less. The same logic applies to the computation of the pre-activations at the last hidden layer and so on, progressing backward through the network. In the limit, if we forced all the weights to be zero, the network would produce a constant output determined by the final bias parameter.

Figure 9.2 shows the effect of fitting the simplified network from figure 8.4 with weight decay and different values of the regularization coeﬀicient λ. When λ is small, it has little effect. However, as λ increases, the fit to the data becomes less accurate, and the function becomes smoother. This might improve the test performance for two reasons:

- If the network is overfitting, then adding the regularization term means that the network must trade off slavish adherence to the data against the desire to be smooth. One way to think about this is that the error due to variance reduces (the model no longer needs to pass through every data point) at the cost of increased bias (the model can only describe smooth functions).
    
- When the network is over-parameterized, some of the extra model capacity de- scribes areas with no training data. Here, the regularization term will favor func- tions that smoothly interpolate between the nearby points. This is reasonable behavior in the absence of knowledge about the true function.

## 9.2 Implicit regularization

An intriguing recent finding is that neither gradient descent nor stochastic gradient descent moves neutrally to the minimum of the loss function; each exhibits a preference for some solutions over others. This is known as implicit regularization.
### 9.2.1 Implicit regularization in gradient descent

Consider a continuous version of gradient descent where the step size is infinitesimal. The change in parameters $\phi$ will be governed by the differential equation:

$$
\frac{\partial \phi}{\partial t} = -\frac{\partial L}{\partial \phi}.
$$

Gradient descent approximates this process with a series of discrete steps of size $\alpha$:
$$
\phi_{t+1} = \phi_t - \alpha \frac{\partial L[\phi_t]}{\partial \phi} ,
$$
(9.7)

The discretization causes a deviation from the continuous path (figure 9.3).

This deviation can be understood by deriving a modified loss term $\tilde{L}$ for the continuous case that arrives at the same place as the discretized version on the original loss $L$. It can be shown (see end of chapter) that this modified loss is:

$$
\tilde{L}_{GD}[\phi] = L[\phi] + \frac{\alpha}{4} \left\| \frac{\partial L}{\partial \phi} \right\|^2 .
$$
(9.8)

In other words, the discrete trajectory is repelled from places where the gradient norm is large (the surface is steep). This doesn’t change the position of the minima where the gradients are zero anyway. However, it changes the effective loss function elsewhere and modifies the optimization trajectory, which potentially converges to a different minimum. Implicit regularization due to gradient descent may be responsible for the observation that full batch gradient descent generalizes better with larger step sizes (figure 9.5a).
