Here, I have implemented Gaussian mixture model by means of Expectation Maximization (EM) algorithm according to the famous C. Bishop's book [1]
The Gaussian mixture probability distribution can be written as a superpositions of multivariate Gaussians:

$$
\begin{align} 
p(x) = \sum\limits_{k=1}^{k} \pi_k\ \mathcal{N}(x|\mu_k,\Sigma_k) 
\end{align}
$$

where $\pi_k$ are mixing coefficients and $\mu_k,\ \Sigma_k$ are mean vectors and covariance matrices respectively.



[1] C. Bishop, Pattern Recognition and Machine Learning (Information Science and Statistics). New York: Springer-Verlag, 2006.
