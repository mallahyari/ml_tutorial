
# Introduction to MLE and MAP

## Maximum Likelihood Estimator (MLE)

## Estimating the Probability of Heads

Let's assume we have a random variable $X$ representing a coin. We are going to estimate the probability that it will turn up heads ($X = 1$) or tails ($X = 0$).

> Task: Estimate the probability of heads $\theta = P(X = 1)$

Evidently, if $P(X=1)=\theta$, then $P(X=0)=1-\theta$. Since we do not know the 'true' probability of heads, i.e. $P(X=1) = \theta$, we will use $\hat\theta$ to refer to it.

**Question:** _What is the probability of $\theta = P(X=1)?$_

In general, _Maximum Likelihood Estimation_ principle asks to choose parameter $\theta$ that maximizes $P(Data|\theta)$, or in other words maximizes the probability of the observed data. We assume that $\theta$ belongs to the set $\Theta \subset \mathbb{R}^n$. Therefore,

$$\hat\theta_{MLE} = \underset{\theta}{\arg\max} P(Data|\theta)$$

In regards to our coin flip example, if we flip the coin repeatedly, we observe that:

- It turns up heads $\alpha_1$ times
- It turns up tails $\alpha_0$ times

Intuitively, we can estimate the $P(X=1)$ from our training data (number of tosses) as the fraction of flips that ends up heads:

$$ P(X=1) = \frac{\alpha_1}{\alpha_1 + \alpha_0}$$

For instance, if we flip the coin 40 times, seeing 18 heads and 22 tails, then we can estimate that:

$$\hat\theta = P(X=1) = \frac{18}{40} = 0.45$$

And if we flip it 5 times, observing 3 heads and 2 tails, then we have:

$$\hat\theta = P(X=1) = \frac{3}{5} = 0.6$$

## How to Calculate MLE?

First step in calculating the maximum likelihood estimator $\hat\theta$ is to define $P(Data|\theta)$. If we flip the coin once, then $P(Data|\theta) = \theta$ if the flip results in heads and $P(Data|\theta) = 1 - \theta$, if the flips turns tails. If we observe $D = \{1,0,1,1,0\}$ by tossing the coin 5 times, assuming the flips are independent and identically distributed (i.i.d), then we have:

$$P(Data|\theta) = \theta\cdot(1-\theta)\cdot\theta\cdot\theta\cdot(1-\theta) = \theta^3\cdot(1-\theta)^2$$

In general, if we flip the coin $n$ times, observing $\alpha_H$ heads and $\alpha_T$ tails, then

$$P(Data|\theta) = \theta^{\alpha_H}\cdot(1-\theta)^{\alpha_T}$$

The next step is to find the value of $\theta$ that maximizes the $P(Data|\theta)$. When finding the MLE, it is often easier to maximize the log-likelihood function since,

$$\underset{\theta}{\arg\max} \log P(Data|\theta) = \underset{\theta}{\arg\max} P(Data|\theta)$$

Let's call $J(\theta) = \log P(Data|\theta)$. Thus, in order to find the value of the $\theta$ that maximizes the $J(\theta)$, we calculate the derivative of $J(\theta)$ with respect to $\theta$, set it to zero and solve for $\theta$.

$$\frac{\partial J(\theta)}{\partial \theta} = \frac{\partial[\alpha_H \log \theta + \alpha_T \log (1-\theta)]}{\partial \theta}= \alpha_H \frac{1}{\theta} - \alpha_T \frac{1}{1-\theta} = 0$$

Solving this for $\theta$ gives, $$\theta = \dfrac{\alpha_H}{\alpha_H + \alpha_T}$$

## Map Estimation for Binomial Distribution

Likelihood is Binomial:$P(Data|\theta)={n\choose \alpha_H}\theta^{\alpha_H}(1-\theta)^{\alpha_T}$

If we assume prior is Beta distribution: $P(\theta)=\frac{\theta^{\beta_H-1}(1-\theta)^\beta_T-1}{B(\beta_H,\beta_T)}\sim Beta(\beta_H, \beta_T)$

 $B(x,y)=\int_o^1 t^{x-1}(1-t)^{y-1}dt$

Then, posterior is Beta distribution: $P(\theta|Data)\sim Beta(\beta_H+\alpha_H, \beta_T+\alpha_T)$


And,

$$
\begin{aligned}
\hat{\theta}_{MAP}&=\underset{\theta}{\arg\max}\ P(Data|\theta) P(\theta)\\
&=\frac{\alpha_H+\beta_H -1}{\alpha_H+\beta_H+\alpha_T+\beta_T -2}
\end{aligned}
$$

- **Conjugate prior:** $P(\theta)$ is the conjugate prior for likelihood function $P(\theta|Data)$ if $P(\theta)$ and $P(\theta|Data)$ have the same form.

- Beta prior is equivalent to extra coin flips

- As the number of samples (e.g. coin flips) increases, the effect of prior is "washed out". It means as $N\rightarrow \infty$, prior is "forgotten".

- For small sample size, prior is important.







