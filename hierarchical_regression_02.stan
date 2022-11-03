//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// Index values, observations, and covariates.
data {
  int<lower = 1> N;               // Number of observations.
  int<lower = 1> K;               // Number of groups.
  int<lower = 1> I;               // Number of observation-level covariates.
  int<lower = 1> J;               // Number of population-level covariates.

  vector[N] y;                    // Vector of observations.
  int<lower = 1, upper = K> g[N]; // Vector of group assignments.
  matrix[N, I] X;                 // Matrix of observation-level covariates.
  matrix[K, J] Z;                 // Matrix of population-level covariates.
}

// Parameters and hyperparameters.
parameters {
  matrix[J, I] Gamma;             // Matrix of population-level coefficients.
  real<lower = 0> tau;            // Variance of the population model.
  matrix[K, I] Beta;              // Matrix of observation-level coefficients.
  real<lower = 0> sigma;          // Variance of the likelihood.
}

// Hierarchical regression.
model {
  // Hyperpriors.
  for (j in 1:J) {
    Gamma[j,] ~ normal(0, 5);
  }
  tau ~ normal(0, 5);

  // Prior.
  sigma ~ normal(0, 5);

  // Population model and likelihood.
  for (k in 1:K) {
    Beta[k,] ~ normal(Z[k,] * Gamma, tau);
  }
  for (n in 1:N) {
    y[n] ~ normal(X[n,] * Beta[g[n],]', sigma);
  }
}

