### Logistic Regression

Logistic regression is a classification technique used for binary classification problems such as classifying tumors as malignant / not malignant, classifying emails as spam / not spam.

## Recap

### Classification

**Classification** is a learning algorithm that determines which discrete category a new example (instance) belongs, given a set of training instances $X$ with their observed categories $Y$. Binary classification is a classification task where $Y$ has two possible values $0,1$. If $Y$ has more than two possible values, it is called a multi-class classification.

### Can we use Linear Regression for classification problems?
Thare are two main issues with using linear regression for classification:

1. Linear regression outputs values for $Y$ that can be much larger than 1 or much lower than 0, but our classes are 0 and 1.

2. Our hypothesis or prediction rule can change each time a new training example arrives, which shouldn't. Instead, we should be able to use the learned hypothesis to make correct predictions for the data we haven't seen before.




### Naive Bayes

If we have consider Naive Bayes assumption, then we have:
$$P(X_1\cdots X_d|Y)=\prod_{i=1}^{d}P(X_i|Y)$$

We also assume parametric form for $P(X_i|Y)$ and $P(Y)$. Then, we use MLE or MAP to estimate the parameters.

At last, Naive Bayes classifier for a $X^{new} = <X_1,X_2,\cdots X_d>$ is:

$$Y^{new} =\underset{y}{\arg\max}\quad P(Y=y_k)\prod_i P(X^{new}_i|Y=y_K)$$

## Generative vs Discriminative Classifiers

Training a classifier involves estimating $P(Y|X)$.

i. Generative classifiers (e.g. Naive Bayes):

- Assume some functional form for $P(X,Y)$, i.e. $P(X|Y)$ and $P(Y)$
- Estimate parameters of $P(X|Y)$, $P(Y)$ directly from training data
- $\underset{y}{\arg\max}\quad P(X|Y)P(Y)= \underset{y}{\arg\max}\quad P(Y|X)$

**Question:** _Can we learn the \$P(Y|X)\$ directly from data? Or better yet, can we learn decision boundary directly?_

ii. Discriminative classifiers (e.g. Logistic Regression):

- Assume some functional form for $P(Y|X)$ or for the decision boundary
- Estimate parameters of $P(Y|X)$ directly from training data

## Logistic Regression

Logistic regression is a classification method for binary classification problems, where input $X$ is a vector of discrete or real-valued variables and $Y$ is discrete (boolean valued). The idea is to learn $P(Y|X)$ directly from observed data.

Let's consider learning $f:X\rightarrow Y$ where,

- $X$ is a vector of real-valued features, $<X_1,\cdots,X_n>$
- $Y$ is boolean
- Assume all $X_i$ are conditionally independent given $Y$
- Assume $P(X_i|Y=y_k) \sim N(\mu_{ik},\sigma_i)$
- Assume $P(Y) \sim $ Bernoulli($\pi$)

What does this imply about the form of $P(Y|X)$?

$$P(Y=0|X=<X_1,\cdots,X_n>)=\frac{1}{1+exp(w_0+\sum_{i}w_iX_i)}$$

$$P(Y=1|X) = 1 - P(Y=1|X) = \frac{\exp(w_0+\sum_{i=1}^{n}w_iX_i)}{1+\exp(w_0+\sum_{i=1}^{n}w_iX_i)} $$

## Logistic Function

In Logistic regression $P(Y|X)$ follows the form of the sigmoid function, which means logistic regression gives the *probability* that an instance belongs to class $1$ or class $0$.

## Logistic Regression is Linear

The reason that logistic regression is linear is that, the outcome is a linear combinations of the inputs and parameters.

$$\frac{P(Y=1|X)}{P(Y=0|X)} = \exp(w_0+\sum_{i=1}^{n}w_iX_i)$$

which implies:

$$\ln\frac{P(Y=1|X)}{P(Y=0|X)} = w_0+\sum_{i=1}^{n}w_iX_i$$

where $w_0+\sum_{i=1}^{n}w_iX_i$ is the linear classification rule.


## Training Logistic Regression: MLCE

We have a collection of training data $D = \{<X^1,Y^1>,\cdots,<X^M,Y^M>\}$. We need to find the parameters $\mathbf{w}=<w_0,\cdots,w_n>$ that **maximize the conditional likelihood** of the training data.

Data likelihood $=\prod_j P(<X^j,Y^j>|\mathbf{w})$, thus data **conditional** likelihood $=\prod_j P(Y^j|X^j,\mathbf{w})$.

therefore,
$$W_{MLCE}=\underset{\mathbf{w}}{\arg\max}\prod_j P(Y^j|X^j,\mathbf{w})$$

In order to make arithmetic easier, we work with the conditional log likelihood. Additionally, we know that maximizing a function is equivalent to *minimizing the negative of the function*. Therefore, we convert our problem to a minimization problem and apply the Gradient Descent algorithm to find the minimum.
$$
\begin{aligned}
  W_{MLCE}&=\underset{\mathbf{w}}{\arg\max}\quad \ln \prod_j P(Y^j|X^j,\mathbf{w}) =  \sum_j \ln P(Y^j|X^j,\mathbf{w})\\
  &=\underset{W}{\arg\max}\quad\sum_j \ln P(Y^j|X^j,\mathbf{w})\\
  &=\underset{W}{\arg\min}\quad -\sum_j \ln P(Y^j|X^j,\mathbf{w})
\end{aligned}
$$

if $J(\mathbf{w})=-\sum_j \ln P(Y^j|X^j,\mathbf{w})$, then we have:

$$\begin{aligned}
  J(\mathbf{w})&=-\sum_j \left [ Y^j \ln P(Y^j=1|X^j,\mathbf{w}) + (1-Y^j) \ln P(Y^j=0|X^j,\mathbf{w})\right ]\\
  &=-\sum_j \left[ Y^j(w_0+\sum_iw_iX_i^j)-\ln(1+\exp(w_0+\sum_iw_iX_i^j))\right ]
\end{aligned}
$$

$J(\mathbf{w})$ is a convex function, so we can always find global minimum. There is no closed-form solution to minimize it. However, we can use gradient descent algorithm to find the minimum.

## Gradient Descent algorithm for Logistic Regression

1. Compute the gradient of $J_D(\mathbf{w})$ over the entire training set $D$:
$$\nabla J_D(\mathbf{w}) = \left [\frac{\partial J_D(\mathbf{w})}{\partial w_0},\cdots \frac{\partial J_D(\mathbf{w})}{\partial w_n} \right]$$

$$\frac{\partial J_D(\mathbf{w})}{\partial w_i}=\sum_{j=1}^{M}X_i^j\left[Y^j-\hat{P}(Y^j=1|X^j,\mathbf{w})\right]$$

We assume $X_0^j=1$, (for $j=0,1,\cdots M$)

2. Do until satisfied:
   - Update the vector of parameters: $w_i=w_i-\eta \frac{\partial J_D(\mathbf{w})}{\partial w_i}$, (for $i=0,1,\cdots n$)

## Using Maximum a Posteriori (MAP) to Estimate Parameters
We assume Gaussian distributions for the prior: $\mathbf{w} \sim N(0,\sigma I)$.Thus,

$$\mathbf{w^*}=\underset{\mathbf{w}}{\arg\max}\quad \ln\left[P(\mathbf{w})\prod_j P(Y^j|X^j,\mathbf{w})\right]$$

Therefore,

$$J(\mathbf{w})=-\sum_j \left[ Y^j \ln P(Y^j=1|X^j,\mathbf{w}) + (1-Y^j) \ln P(Y^j=0|X^j,\mathbf{w})\right]+\lambda\sum_{i=1}^{n}w_i^2$$

where $\lambda \sum_{i=1}^{n}w_i^2$ is called a **regularization** term. Regularization helps reduce the overfitting, and also keeps the weights near to zero (i.e. discourage the weights from getting large values).


And the modified gradient descent rule is:

$$w_i=w_i-\eta \left [\frac{\partial J_D(\mathbf{w})}{\partial w_i} + \lambda w_i \right ]$$










<!--
## How to Derive $P(Y|X)$ for Gaussian $P(X_i|Y=y_k)$ assuming $\sigma_{ik}=\sigma_i$

$$
\begin{aligned}
P(Y=1|X) &= \frac{P(Y=1)P(X|Y=1)}{P(Y=1)P(X|Y=1) + P(Y=0)P(X|Y=0)}\\\\
&= \frac{1}{1+\dfrac{P(Y=0)P(X|Y=0)}{P(Y=1)P(X|Y=1)}}\\\\
&=\frac{1}{1+\exp(\ln \left ( \dfrac{P(Y=0)P(X|Y=0)}{P(Y=1)P(X|Y=1)} \right)}\\\\
&= \frac{1}{1+\exp\left ((\ln \dfrac{1-\pi}{\pi}) + (\sum_{i}\ln \dfrac{P(X_i|Y=0)}{P(X_i|Y=1)}) \right)}
\end{aligned}
$$

We know that $P(x|y_k)=\frac{1}{\sigma_{ik}\sqrt{2\pi}}e^{\frac{-(x-\mu_{ik})^2}{2\sigma_{ik}^2}}$, thus:

$$P(Y=1|X)=\frac{1}{1+\exp(w_0+\sum_{i=1}^{n}w_iX_i)}$$

where $w_i=\frac{\mu_{i0}-\mu_{i1}}{\sigma_i^2}$ for all $i=1,\cdots,n$

For all the details, please see [Tom Mitchell's Chapter on Logistic Regression]("http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf").


We can compute $P(Y=0|X)$ from $P(Y=1|X)$ as follows:

$$P(Y=0|X) = 1 - P(Y=1|X) = \frac{\exp(w_0+\sum_{i=1}^{n}w_iX_i)}{1+\exp(w_0+\sum_{i=1}^{n}w_iX_i)} $$ -->



### References
1. https://stats.stackexchange.com/questions/22381/why-not-approach-classification-through-regression
2.https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html











