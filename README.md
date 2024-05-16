## Gaussian mixture model

Here, I have implemented Gaussian mixture model by means of Expectation Maximization (EM) algorithm according to the famous $\textit {C. Bishop's book}$ [1].
The Gaussian mixture probability distribution can be written as a superposition of multivariate Gaussians:

$$
\begin{align} 
p(x) = \sum\limits_{k=1}^{K} \pi_k\ \mathcal{N}(x|\mu_k,\Sigma_k) 
\end{align}
$$

where $\sum\limits_{k=1}^{K}\pi_k = 1$ are mixing coefficients and $\mu_k,\ \Sigma_k$ are mean vectors and covariance matrices of Gaussian constituents. Our task is given the observations $X = {x_1, x_2,..., x_N}$, infer the parameters $\pi_k,\ \mu_k,\ \Sigma_k$.

To do this, we treat the Gaussian constituents as binary latent variables $Z = {z_1,..., z_n}$ where $z_k$ is a $K$-dimensional vector where only one component is equal to 1 and others are 0. The graphical model is presented below

$\displaylines{}$ 
<p align="center">
<img src = "https://github.com/baronett90210/Gaussian-mixture-model/assets/136889949/82c15f64-f45f-425d-84d3-4f74fb9b742a" width="250" height="200">
</p>


Now, we can formulate the joint likelyhood function: 

$$
\begin{align} 
p(X, Z|**\pi**,\ **\mu**,\ **\Sigma**) = \prod_{n=1}^{N}\prod_{k=1}^{K} \pi_k^{z_{nk}}\ \mathcal{N}(x_n|\mu_k,\Sigma_k)^{z_{nk}} 
\end{align}
$$

where binary $z_{nk}\in 0, 1$ denote a $k$ component of a vector $z_n$.

Next, the likelyhood function is maximized iteratively w.r.t. parametres $\pi_k,\ \mu_k,\ \Sigma_k$. At each iteration, there are E-step and M-step. The former evaluates the expected value of the log-likelyhood w.r.t. the latent variables:

$$
\begin{align} 
\mathbb{E_**Z**}(ln\ p(X, Z|**\pi**,\ **\mu**,\ **\Sigma**)) =  \sum_{n=1}^{N}\sum_{k=1}^{K} \gamma(z_{nk})(ln\ \pi_k + ln\ \mathcal{N}(x_n|\mu_k,\Sigma_k) )
\end{align}
$$

$$
\begin{align} 
\gamma(z_{nk})=\frac{p(x_n|z_{nk}=1) p(z_{nk}=1)} {\sum\limits_{j=1}^{K} p(x_n|z_{nj} = 1)p(z_{nj} = 1)} = \frac{\pi_k\ \mathcal{N}(x_n|\mu_k,\Sigma_k)}{\sum\limits_{j=1}^{K} \pi_j\ \mathcal{N}(x_n|\mu_j,\Sigma_j)}
\end{align}
$$

Here, $\gamma(z_{nk})$ denotes the responsibility of a Gaussian constituent $k$ for a data point $n$. The M-step maximizes the expected value w.r.t. the parameteres which leads to the following iterative updates: 

$$
\begin{align} 
\mu_k^{new} = \frac{1}{N_k}\sum\limits_{n=1}^{N} {\gamma(z_{nk}) x_n} 
\end{align}
$$

$$
\begin{align}
\Sigma_k^{new} = \frac{1}{N_k}\sum\limits_{n=1}^{N}\gamma(z_{nk})(x_n-\mu_k^{new})(x_n-\mu_k^{new})^T 
\end{align}
$$

$$
\begin{align}
\pi_k^{new} = \frac{N_k}{N}
\end{align}
$$

where $N_k=\sum\limits_{n=1}^{N}\gamma(z_{nk})$ - expected number of points assigned to kth cluster.

To run the code example use $\bf{GMM.py}$. There we first generate random artificial data from 3 Gaussian distributions with the following parameters: 

$$
\mu = \begin{vmatrix}
1 & 2\\
-1 & -1\\
-1.5 & 2
\end{vmatrix}
$$

$$
\Sigma = \begin{vmatrix}
0.5 & -0.5\\
-0.5 & 1
\end{vmatrix}
\begin{vmatrix}
0.5 & 0.2\\
0.2 & 0.5
\end{vmatrix}
\begin{vmatrix}
0.7 & 0.35\\
0.35 & 0.3
\end{vmatrix}
$$

$$
\begin{align}
\pi = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3} )
\end{align}
$$

Applying the EM algorithm with $K = 3$ we manage to recover the parameteres. Below you can monitor how the solutions change with iterations together with the log-likelyhood function.

$\displaylines{}$ 
<p align="center">
<img src = "https://github.com/baronett90210/Gaussian-mixture-model/assets/136889949/382cf101-695b-463e-830e-82272c133f0c" width="900" height="500">
</p>

## Variational Gaussian mixture 

Now, we are moving towards probabilistically driven Bayesian machine learning. Essentially, we should consider more realistic situation where all the parameters are random variables. So the model is described by the joint probability:

$$
\begin{align}
P(X, Z, \pi, \mu, \Lambda) = P(X| Z, \mu, \Lambda) P(Z| \pi) P(\mu| \Lambda) P(\Lambda) P(\pi)
\end{align}
$$

Note that we have established some conditional relationship between the variables. Let us consider a graph to understand them:

$\displaylines{}$ 
<p align="center">
<img src = "https://github.com/baronett90210/Gaussian-mixture-model/assets/136889949/901eb3df-b938-42ec-b098-f4143ea2a113" width="250" height="200">
</p>

The mixing coefficients $\pi$ follows Dirichlet distribution $P(\pi| \alpha_0) = C(\alpha_0) \prod\limits_{k}^{K} \pi_{k}^{\alpha_{0}-1}$ where $\alpha_0$ represents the number of observations per mixture. The indicator variable $Z$ follows the multinomial distribution $P(Z| \pi) = \prod\limits_{n=1}^{N}\prod\limits_{k=1}^{K} \pi_k^{z_{nk}}$. The observed data $X$ is written similarly as before $P(X| Z, \mu, \Lambda)= \prod\limits_{n=1}^{N}\prod\limits_{k=1}^{K} \mathcal{N}(x_n|\mu_k,\Lambda_k^{-1})^{z_{nk}}$ but now means and covariance matrices of Gaussian constituents are random variables themselves. Conventionally $P(\mu, \Lambda) = P(\mu| \Lambda) P(\Lambda) = \prod\limits_{k}^{K} \mathcal{N}(\mu_k|0, (\beta_0 \Lambda_k)^{-1})\ W(\Lambda_k| W_0, \nu_0)$ is Gaussian-Wishart distribution.

Next, one can use a factorized approximation of Variational Inference to derive the iterative optimization algorithm. The general idea is to seek an approximation $Q(Y)$ of the true posterior distribution $P(Y|X)$ of all latent variables $Y$. The approximation is achieved by restricting the distribution family to factorized ones, i.e. $Q(Y) = \prod\limits_{k}^{K}Q_k(Y_k)$. Using this one obtains the update rule $lnQ_{j}^{*}(Z_{j}) = \mathbb{E_{i\neq j}}(ln\ P(X, Y)) + Const$, so the logarithm of the optimum latent distribution $Q_j(Z_j)$ is equal to the expectation of the total join distribution w.r.t. all latent parameter except $Z_j$. Convenietly, this update can be applied for $Q_j(Z_j)$ one by one. We will note provide the full update equations for our particular case of the Variational Gaussian mixture, an interested reader is refered to [1]. 

Finally, let us check Variational Inference out! To run, use $\bf{GMM\ VB.py}$. As usually we generate the data from 3 Gaussian distributions. Interestingly, the variational model doesn't need to know the true number of constituents $K$. We just start with a sufficient number (let's say 10), so initially we generate 10 Gaussians, all with zero means. The figure below show the dynamics of the iterations together with lower bound (ELBO). Note the discrete jumps in ELBO plot. This happens when the expected value of the mixing coefficients drops below a small threshold, that we put in the code. Remarkably, in the end the algorithm finds 3 mixtures by itself, getting rid of the probabilistically negligible consituents. 

$\displaylines{}$ 
<p align="center">
<img src = "https://github.com/baronett90210/Gaussian-mixture-model/assets/136889949/249bb401-127a-44f5-9971-b9daba0e8ceb" width="900" height="500">
</p>

$\displaylines{}$ 
<p align="center">
<img src = "https://github.com/baronett90210/Gaussian-mixture-model/assets/136889949/426e0c42-141b-4e0e-af1e-6bdd45001d2a" width="500" height="350">
</p>


[1] C. Bishop, Pattern Recognition and Machine Learning (Information Science and Statistics). New York: Springer-Verlag, 2006.
