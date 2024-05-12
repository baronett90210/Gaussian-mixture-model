Here, I have implemented Gaussian mixture model by means of Expectation Maximization (EM) algorithm according to the famous C. Bishop's book [1]
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
<img src = "https://github.com/baronett90210/Gaussian-mixture-model/assets/136889949/901eb3df-b938-42ec-b098-f4143ea2a113" width="250" height="200">
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

[1] C. Bishop, Pattern Recognition and Machine Learning (Information Science and Statistics). New York: Springer-Verlag, 2006.
