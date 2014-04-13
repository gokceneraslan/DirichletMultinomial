DMN.cluster <- function(count.data, K, seed=F, verbose=F, eta=0.1, nu=0.1, EM.maxit=250,
                        EM.threshold=1e-6, soft.kmeans.maxit=1000,
                        soft.kmeans.stiffness=50, randomInit=T) {

  if (seed != F) set.seed(seed)

  N <- nrow(count.data)
  S <- ncol(count.data)

  kmeans.res <- soft_kmeans(count.data, K, verbose=verbose,
                            randomInit=randomInit, stiffness=soft.kmeans.stiffness)

  alpha <- kmeans.res$centers
  alpha[alpha <= 0] <- 1e-6
  lambda <- log(alpha)
  Ez <- kmeans.res$labels
  weights <- rowSums(Ez)

  if (verbose)
    cat('Expectation Maximization setup\n')

  for (k in 1:K) {
    lambda[k,] = optimise_lambda_k(lambda[k,], count.data, Ez[k,], eta, nu);
  }

  if (verbose)
    cat('Expectation Maximization\n')

  #EM loop
  iter <- 0
  last.nll <- 0
  nll.change <- .Machine$double.xmax

  while ((iter < EM.maxit) && (nll.change > EM.threshold)) {

    if (verbose)
      print('Calculating Ez values...')

    Ez <- calc_z(Ez, count.data, weights, lambda)

    if (verbose)
      print('Optimizing lambda...')

    for (k in 1:K) {
      lambda[k,] <- optimise_lambda_k(lambda[k,], count.data, Ez[k,], eta, nu);
    }

    weights <- rowSums(Ez)

    if (verbose)
      print('Calculating negative log likelihood...')

    #calculate neg. likelihood for convergence
    nll <- neg_log_likelihood(weights, lambda, count.data, eta, nu);
    nll.change <- abs(last.nll - nll)
    last.nll <- nll

    iter <- iter+1

    if (verbose)
      print(paste('--> EM Iteration:', iter, 'Neg.LL change:', round(nll.change, 6)))
  }

  # hessian
  if (verbose)
    cat("  Hessian\n");

  err <- matrix(0, K, S)
  logDet <- 0

  for (k in 1:K) {
    if (k > 0)
      logDet <- logDet + 2.0 * log(N) - log(weights[k])

    hess <- hessian(lambda[k, ], Ez[k, ], count.data, nu)
    invHess <- solve(hess) #inverse of Hessian matrix
    err[k, ] <- diag(invHess)
    logDet <- logDet + sum(log(abs(diag(hess))))
  }

  P <- K*S+K-1
  gof.laplace <- last.nll + 0.5 * logDet - 0.5 * P * log(2.0 * pi);
  gof.BIC <- last.nll + 0.5 * log(N) * P
  gof.AIC <- last.nll + P
  gof <- c(NLE=last.nll, LogDet=logDet, Laplace=gof.laplace, AIC=gof.AIC, BIC=gof.BIC)

  result <- list()

  result$GoodnessOfFit <- gof
  result$Group <- t(Ez)
  mixture_list <- mixture_output(count.data, weights, lambda, err)
  result$Mixture <- list(Weight=mixture_list$Mixture)

  result$Fit <- list(Estimate=t(mixture_list$Estimate),
                     Upper=t(mixture_list$Upper),
                     Lower=t(mixture_list$Lower))

  return(result)
}