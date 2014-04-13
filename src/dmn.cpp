// [[Rcpp::depends(RcppGSL)]]

#include <gsl/gsl_vector.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_permutation.h>

#include <Rcpp.h>
#include <RcppGSL.h>
using namespace Rcpp;

#define BIG_DBL 1.0e9
#define MAX_GRAD_ITER 1000

// [[Rcpp::export]]
List soft_kmeans(IntegerMatrix data, int K, bool verbose=false,
            bool randomInit=true, double stiffness=50.0,
            double threshold=1.0e-6, int maxIt=1000) {

    const int S = data.ncol(), N = data.nrow();
    int i, j, k, iter = 0;

    NumericVector W(K);
    NumericMatrix Z(K, N);
    List data_dimnames = data.attr("dimnames");
    Z.attr("dimnames") = List::create(CharacterVector(), data_dimnames[0]);
    NumericMatrix Mu(K, S);
    Mu.attr("dimnames") = List::create(CharacterVector(), data_dimnames[1]);

    double dMaxChange = BIG_DBL;

    if (verbose)
        Rprintf("  Soft kmeans\n");

    NumericMatrix Y(N, S);
    NumericVector Mu_row(S);

    for (i = 0; i < N; i++) {
        int iTotal = 0;
        for (j = 0; j < S; j++)
            iTotal += data(i, j);

        if (iTotal == 0) iTotal = 1; //workaround for -nan error

        for (j = 0; j < S; j++)
            Y(i,j) = (data(i, j)) / (double)iTotal;
    }

    /* initialise */
    for (i = 0; i < N; i++) {
        if (K == 1)
            k = 0;
        else if (randomInit)
            k = round(Rcpp::as<double>(runif(1))*(K-1)); //random integer btw 1 and K
        else
            k = i % K;

        Z(k, i) = 1.0;
    }

    while (dMaxChange > threshold && iter < maxIt) {
        /* update mu */
        dMaxChange = 0.0;
        for (i = 0; i < K; i++){
            double dNormChange = 0.0;
            W[i] = 0.0;
            for (j = 0; j < N; j++)
                W[i] += Z(i,j);

            for (j = 0; j < S; j++) {
                Mu_row[j] = 0.0;
                for (k = 0; k < N; k++)
                    Mu_row[j] += Z(i, k)*Y(k, j);
            }

            for (j = 0; j < S; j++) {
                double dDiff = 0.0;
                Mu_row[j] /= W[i];

                dDiff = (Mu_row[j] - Mu(i, j));
                dNormChange += dDiff * dDiff;
                Mu(i,j) = Mu_row[j];
            }
            dNormChange = sqrt(dNormChange);
            if (dNormChange > dMaxChange)
                dMaxChange = dNormChange;
        }

        /* calc distances and update Z */
        for (i = 0; i < N; i++) {
            double dNorm = 0.0, adDist[K];
            for (k = 0; k < K; k++) {
                adDist[k] = 0.0;
                for (j = 0; j < S; j++) {
                    const double dDiff = (Mu(k, j) - Y(i, j));
                    adDist[k] += dDiff * dDiff;
                }
                adDist[k] = sqrt(adDist[k]);
                dNorm += exp(-stiffness * adDist[k]);
            }
            for (k = 0; k < K; k++)
                Z(k, i) = exp(-stiffness * adDist[k]) / dNorm;
        }
        iter++;

        if (verbose && (iter % 10 == 0))
            Rprintf("    iteration %d change %f\n", iter, dMaxChange);
    }

    return List::create(_["centers"] = Mu,
                        _["weights"] = W,
                        _["labels"] = Z);
}


static double neg_log_evidence_lambda_pi(const gsl_vector *lambda,
        void *params)
{
    int i, j;

    List lparams = wrap((SEXP) params);
    IntegerMatrix aanX = lparams["data"];
    NumericVector adPi = lparams["pi"];
    double GAMMA_ITA = lparams["eta"];
    double GAMMA_NU = lparams["nu"];

    const int S = aanX.ncol(), N = aanX.nrow();

    double dLogE = 0.0, dLogEAlpha = 0.0, dSumAlpha = 0.0, dSumLambda = 0.0;
    double adSumAlphaN[N], dWeight = 0.0;

    for (i = 0; i < N; i++) {
        adSumAlphaN[i] = 0.0;
        dWeight += adPi[i];
    }

    for (j = 0; j < S; j++) {
        const double dLambda = gsl_vector_get(lambda, j);
        const double dAlpha = exp(dLambda);
        dLogEAlpha += gsl_sf_lngamma(dAlpha);
        dSumLambda += dLambda;
        dSumAlpha += dAlpha;
        const double lngammaAlpha0 = gsl_sf_lngamma(dAlpha);
        for (i = 0; i < N; i++) {
            const double dN = aanX(i, j);
            const double dAlphaN = dAlpha + dN;
            const double lngammaAlphaN = dN ? gsl_sf_lngamma(dAlphaN) : lngammaAlpha0;
            adSumAlphaN[i] += dAlphaN; //weight by pi
            dLogE -= adPi[i] * lngammaAlphaN; //weight by pi
        }
    }
    dLogEAlpha -= gsl_sf_lngamma(dSumAlpha);

    for(i = 0; i < N; i++)
        dLogE += adPi[i] * gsl_sf_lngamma(adSumAlphaN[i]);

    return dLogE + dWeight*dLogEAlpha + GAMMA_NU*dSumAlpha -
        GAMMA_ITA * dSumLambda;
}

static void neg_log_derive_evidence_lambda_pi(const gsl_vector *ptLambda,
        void *params, gsl_vector* g)
{
    //const struct data_t *data = (const struct data_t *) params;
    //const int S = data->S, N = data->N, *aanX = data->aanX;
    //const double *adPi = data->adPi;

    List lparams = wrap((SEXP) params);
    IntegerMatrix aanX = lparams["data"];
    NumericVector adPi = lparams["pi"];
    double GAMMA_ITA = lparams["eta"];
    double GAMMA_NU = lparams["nu"];

    const int S = aanX.ncol(), N = aanX.nrow();

    int i, j;
    double adDeriv[S], adStore[N], adAlpha[S];
    double dSumStore = 0.0, dStore = 0.0;
    double dWeight = 0;

    for (i = 0; i < N; i++) {
        adStore[i] = 0.0;
        dWeight += adPi[i];
    }

    for (j = 0; j < S; j++) {
        adAlpha[j] = exp(gsl_vector_get(ptLambda, j));
        dStore += adAlpha[j];
        adDeriv[j] = dWeight* gsl_sf_psi(adAlpha[j]);
        double alphaS0 = gsl_sf_psi(adAlpha[j]);
        for (i = 0; i < N; i++) {
            int dN = aanX(i, j);
            double dAlphaN = adAlpha[j] + dN;

            double psiAlphaN = dN ? gsl_sf_psi(dAlphaN) : alphaS0;
            adDeriv[j] -= adPi[i]*psiAlphaN;
            //            adDeriv[j] -= adPi[i]*gsl_sf_psi (dAlphaN);
            adStore[i] += dAlphaN;
        }
    }

    for (i = 0; i < N; i++)
        dSumStore += adPi[i] * gsl_sf_psi(adStore[i]);
    dStore = dWeight * gsl_sf_psi(dStore);

    for (j = 0; j < S; j++) {
        double value = adAlpha[j] *
            (GAMMA_NU + adDeriv[j] - dStore + dSumStore) - GAMMA_ITA;
        gsl_vector_set(g, j, value);
    }
}

static void neg_log_FDF_lamba_pi(const gsl_vector *x, void *params,
        double *f, gsl_vector *g)
{
    *f = neg_log_evidence_lambda_pi(x, params);
    neg_log_derive_evidence_lambda_pi(x, params, g);
}


// [[Rcpp::export]]
NumericVector optimise_lambda_k(NumericVector adLambdaK, IntegerMatrix data,
                              NumericVector adZ, double eta, double nu)
{
    const int S = data.ncol();

    int i, status;
    size_t iter = 0;

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;
    gsl_multimin_function_fdf fdf;
    gsl_vector *ptLambda;

    //initialise vector
    ptLambda = gsl_vector_alloc(S);
    for (i = 0; i < S; i++)
        gsl_vector_set(ptLambda, i, adLambdaK[i]);

    //initialise function to be solved
    List params = List::create(_["pi"] = adZ, _["data"] = data,
                               _["eta"] = eta, _["nu"] = nu);
    fdf.n = S;
    fdf.f = neg_log_evidence_lambda_pi;
    fdf.df = neg_log_derive_evidence_lambda_pi;
    fdf.fdf = neg_log_FDF_lamba_pi;
    fdf.params = params;

    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, S);

    gsl_multimin_fdfminimizer_set(s, &fdf, ptLambda, 1.0e-6, 0.1);

    do {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);
        if (status)
            break;
        status = gsl_multimin_test_gradient(s->gradient, 1e-3);
    } while (status == GSL_CONTINUE && iter < MAX_GRAD_ITER);

    for (i = 0; i < S; i++)
        adLambdaK[i] = gsl_vector_get(s->x, i);

    gsl_vector_free(ptLambda);
    gsl_multimin_fdfminimizer_free(s);

    return adLambdaK;
}

// [[Rcpp::export]]
double neg_log_evidence_i(IntegerMatrix data, int rowNo, NumericVector Lambda,
                          NumericVector LnGammaLambda0)
{
    int j;
    const int S = data.ncol(), N = data.ncol();
    double dLogE = 0.0, dLogEAlpha = 0.0, dSumAlpha = 0.0,
           dSumAlphaN = 0.0;

    for (j = 0; j < S; j++) {
        const double n = data(rowNo, j);
        const double dAlpha = exp(Lambda[j]);
        const double dAlphaN = n + dAlpha;

        dLogEAlpha += LnGammaLambda0[j];
        dSumAlpha += dAlpha;
        dSumAlphaN += dAlphaN;
        dLogE -= n ? gsl_sf_lngamma(dAlphaN) : LnGammaLambda0[j] ;
    }

    dLogEAlpha -= gsl_sf_lngamma(dSumAlpha);
    dLogE += gsl_sf_lngamma(dSumAlphaN);

    return dLogE + dLogEAlpha;
}

// [[Rcpp::export]]
NumericMatrix calc_z(NumericMatrix Z, IntegerMatrix data,
                     NumericVector W, NumericMatrix Lambda)
{
    int i, j, k;
    const int N = data.nrow(), S = data.ncol(), K = W.length();
    double adStore[K];
    NumericMatrix LngammaLambda0(K, S);

    for(k = 0; k < K; k++){
        for(j = 0; j < S; j++){
            const double dAlpha = exp(Lambda(k, j));
            LngammaLambda0(k, j) = gsl_sf_lngamma(dAlpha);
        }
    }

    for (i = 0; i < N; i ++) {
        double dSum = 0.0;
        double dOffset = BIG_DBL;
        for (k = 0; k < K; k++) {
            double dNegLogEviI =
                neg_log_evidence_i(data, i, Lambda(k, _), LngammaLambda0(k, _));

            if (dNegLogEviI < dOffset)
                dOffset = dNegLogEviI;
            adStore[k] = dNegLogEviI;
        }
        for (k = 0; k < K; k++) {
            Z(k, i) = W[k] * exp(-(adStore[k] - dOffset));
            dSum += Z(k, i);
        }
        for (k = 0; k < K; k++)
            Z(k, i) /= dSum;
    }
  return Z;
}

// [[Rcpp::export]]
double neg_log_likelihood(NumericVector W, NumericMatrix Lambda,
                          IntegerMatrix data, double eta, double nu)
{
    const int S = data.ncol(), N = data.nrow(), K = W.length();

    int i, j, k;
    double adPi[K], adLogBAlpha[K];
    double dRet = 0.0, dL5 = 0.0, dL6 = 0.0, dL7 = 0.0, dL8 = 0.0;
    double GAMMA_ITA = eta, GAMMA_NU = nu;

    NumericMatrix LngammaLambda0(K, S);
    for (k = 0; k < K; k++){
        double dSumAlphaK = 0.0;
        adLogBAlpha[k] = 0.0;
        adPi[k] = W[k]/N;
        for (j = 0; j < S; j++){
            double dAlpha = exp(Lambda(k, j));
            double lngammaAlpha = gsl_sf_lngamma(dAlpha);
            LngammaLambda0(k, j) = lngammaAlpha;

            dSumAlphaK += dAlpha;
            adLogBAlpha[k] += lngammaAlpha;
        }
        adLogBAlpha[k] -= gsl_sf_lngamma(dSumAlphaK);
    }
    for (i = 0; i < N; i++) {
        double dProb = 0.0, dFactor = 0.0, dSum = 0.0, adLogStore[K],
               dOffset = -BIG_DBL;

        for (j = 0; j < S; j++) {
            dSum += data(i, j);
            dFactor += gsl_sf_lngamma(data(i, j) + 1.0);
        }
        dFactor -= gsl_sf_lngamma(dSum + 1.0);

        for (k = 0; k < K; k++) {
            double dSumAlphaKN = 0.0, dLogBAlphaN = 0.0;
            for (j = 0; j < S; j++) {
                int countN = data(i, j);
                double dAlphaN = exp(Lambda(k, j)) + countN;
                dSumAlphaKN += dAlphaN;
                dLogBAlphaN += countN ? gsl_sf_lngamma(dAlphaN) : LngammaLambda0(k, j);
            }
            dLogBAlphaN -= gsl_sf_lngamma(dSumAlphaKN);
            adLogStore[k] = dLogBAlphaN - adLogBAlpha[k] - dFactor;
            if (adLogStore[k] > dOffset)
                dOffset = adLogStore[k];
        }

        for (k = 0; k < K; k++)
            dProb += adPi[k]*exp(-dOffset + adLogStore[k]);
        dRet += log(dProb)+dOffset;
    }
    dL5 = -S * K * gsl_sf_lngamma(GAMMA_ITA);
    dL6 = GAMMA_ITA * K * S * log(GAMMA_NU);
    for (i = 0; i < K; i++)
        for (j = 0; j < S; j++) {
            dL7 += exp(Lambda(i, j));
            dL8 += Lambda(i, j);
        }
    dL7 *= -GAMMA_NU;
    dL8 *= GAMMA_ITA;
    return -dRet -dL5 - dL6 -dL7 -dL8;
}

// [[Rcpp::export]]
NumericMatrix hessian(NumericVector Lambda, NumericVector Pi,
                      IntegerMatrix data, double nu)
{
    const int S = data.ncol(), N = data.nrow();
    NumericMatrix Hessian(S, S);

    int i = 0, j = 0;
    double adAlpha[S], adAJK[S], adCJK[S], adAJK0[S], adCJK0[S];
    double dCK0 = 0.0, dAK0;
    double dCSum, dAlphaSum = 0.0, dW = 0.0, dCK = 0.0, dAK;

    for (j = 0; j < S; j++) {
        adAlpha[j] = exp(Lambda[j]);
        dAlphaSum += adAlpha[j];
        adAJK0[j] = adAJK[j] = adCJK0[j] = adCJK[j] = 0.0;
        const double dPsiAlpha = gsl_sf_psi(adAlpha[j]);
        const double dPsi1Alpha = gsl_sf_psi_1(adAlpha[j]);
        for (i = 0; i < N; i++) {
            const int n = data(i, j);
            adCJK0[j] += Pi[i] * n ? gsl_sf_psi(adAlpha[j] + n) : dPsiAlpha;
            adAJK0[j] += Pi[i] * dPsiAlpha;
            adCJK[j] += Pi[i] * n ? gsl_sf_psi_1(adAlpha[j] + n): dPsi1Alpha;
            adAJK[j] += Pi[i] * dPsi1Alpha;
        }
    }

    for (i = 0; i < N; i++) {
        dW += Pi[i];
        dCSum = 0.0;
        for (j = 0; j < S; j++)
            dCSum += adAlpha[j] + data(i, j);
        dCK  += Pi[i]*gsl_sf_psi_1(dCSum);
        dCK0 += Pi[i]*gsl_sf_psi(dCSum);
    }

    dAK = dW * gsl_sf_psi_1(dAlphaSum);
    dAK0 = dW * gsl_sf_psi(dAlphaSum);
    for (i = 0; i < S; i++)
        for (j = 0; j < S; j++) {
            double dVal = 0.0;
            if (i == j) {
                double dG1 = -adAlpha[i] *
                    (dAK0 - dCK0 + adCJK0[i] - adAJK0[i]);
                double dG2 = -adAlpha[i] *
                    adAlpha[i]*(dAK - dCK + adCJK[i] - adAJK[i]);
                double dG3 = adAlpha[i] * nu;
                dVal = dG1 + dG2 + dG3;
            } else
                dVal = -adAlpha[i] * adAlpha[j] * (dAK - dCK);
            Hessian(i, j) = dVal;
        }
    return Hessian;
}
/*
static void group_output(struct data_t *data, double** aadZ)
{
    const int N = data->N, K = data->K;
    int i, k;
    for(k = 0; k < K; k++)
        for (i = 0; i < N; i++)
            data->group[k * N + i] = aadZ[k][i];
}
*/

// [[Rcpp::export]]
List mixture_output(IntegerMatrix data, NumericVector W,
                    NumericMatrix Lambda, NumericMatrix Err)
{
    const int N = data.nrow(), S = data.ncol(), K = W.length();
    int i, k;

    NumericVector mixture_wt(K);
    NumericMatrix fit_lower(K, S), fit_upper(K, S), fit_mpe(K, S);

    fit_lower.attr("dimnames") = Lambda.attr("dimnames");
    fit_upper.attr("dimnames") = Lambda.attr("dimnames");
    fit_mpe.attr("dimnames") = Lambda.attr("dimnames");

    for (k = 0; k < K; k++)
        mixture_wt[k] = W[k] / N;

    for (i = 0; i < S; i++) {
        for (k = 0; k < K; k++) {
            double dErr = Err(k, i), dL = 0.0, dU = 0.0;
            int bIll = FALSE;
            if (dErr >= 0.0) {
                dErr = sqrt(dErr);
                if (dErr < 100.0) {
                    dL =  exp(Lambda(k, i) - 2.0*dErr);
                    dU =  exp(Lambda(k, i) + 2.0*dErr);
                } else bIll = TRUE;
            } else bIll = TRUE;

            if (bIll)
                dL = dU = R_NaN;
            fit_lower(k, i) = dL;
            fit_mpe(k, i) = exp(Lambda(k, i));
            fit_upper(k, i) = dU;
        }
    }
    return List::create(_["Lower"]=fit_lower,
                        _["Upper"]=fit_upper,
                        _["Estimate"]=fit_mpe,
                        _["Mixture"]=mixture_wt);
}

/*
// [[Rcpp::export]]
List dirichlet_fit(IntegerMatrix data, int K, int seed = -1,
                        int maxIt=250, bool verbose=true,
                        double eta=0.1, double nu=0.1, double stiffness=50.0,
                        bool randomInit=true)
{
    const int N = data.nrow(), S = data.ncol();
    int i, j, k;
    RNGScope rng; //initialize RNG

    if(seed != -1) {
        Function setseed("set.seed");
        setseed(seed);
    }

    NumericMatrix Z(K, N), Lambda(K, S), Err(K, S);
    NumericVector W(K);

    // soft k means initialiser
    List kmeans_result = soft_kmeans(data, K, verbose,
                                     randomInit, stiffness);
    Lambda = as<NumericMatrix>(kmeans_result["centers"]);
    //W = kmeans_result["weights"];
    Z = as<NumericMatrix>(kmeans_result["labels"]);

    for (k = 0; k < K; k++) {
        for (i = 0; i < N; i++)
            W[k] += Z(k, i);
    }

    if (verbose)
        Rprintf("  Expectation Maximization setup\n");

    for (k = 0; k < K; k++) {
        for (j = 0; j < S; j++) {
            const double x = Lambda(k, j);
            Lambda(k, j) = (x > 0.0) ? log(x) : -10;
        }
        Lambda(k, _) = optimise_lambda_k(Lambda(k, _), data, Z(k, _), eta, nu);
    }

    // simple EM algorithm
    int iter = 0;
    double dNLL = 0.0, dNew = 0.0, dChange = BIG_DBL;

    if (verbose)
        Rprintf("  Expectation Maximization\n");

    while (dChange > 1.0e-6 && iter < maxIt) {
        if (verbose)
            Rprintf("  Calculating Ez...\n");

        Z = calc_z(Z, data, W, Lambda); // latent var expectation

        if (verbose)
            Rprintf("  Optimizing wrt Lambda values...\n");

        for (k = 0; k < K; k++) // mixture components, given pi
            Lambda(k, _) = optimise_lambda_k(Lambda(k, _), data, Z(k, _), eta, nu);

        for (k = 0; k < K; k++) { // current likelihood & weights
            W[k] = 0.0;
            for(i = 0; i < N; i++)
                W[k] += Z(k, i);
        }

        if (verbose)
            Rprintf("  Calculate negative loglikelihood...\n");

        dNew = neg_log_likelihood(W, Lambda, data, eta, nu);
        dChange = fabs(dNLL - dNew);
        dNLL = dNew;
        iter++;
        checkUserInterrupt();
        if (verbose && (iter % 1) == 0)
            Rprintf("    iteration %d change %.6f\n", iter, dChange);
    }

    // hessian
    if (verbose)
        Rprintf("  Hessian\n");

    gsl_permutation *p = gsl_permutation_alloc(S);
    double dLogDet = 0.0, dTemp;
    int signum, status;

    for (k = 0; k < K; k++) {
        if (k > 0)
            dLogDet += 2.0 * log(N) - log(W[k]);

        RcppGSL::matrix<double> Hessian(hessian(Lambda(k, _), Z(k, _), data, nu));
        RcppGSL::matrix<double> InverseHessian(S, S);

        status = gsl_linalg_LU_decomp(Hessian, p, &signum);
        gsl_linalg_LU_invert(Hessian, p, InverseHessian);
        for (j = 0; j < S; j++) {
            Err(k, j) = InverseHessian(j, j);
            dTemp = Hessian(j, j);
            dLogDet += log(fabs(dTemp));
        }
        Hessian.free();
        InverseHessian.free();
    }
    gsl_permutation_free(p);

    // results
    List result;
    double dP = K * S + K - 1;
    double laplace = dNLL + 0.5 * dLogDet - 0.5 * dP * log(2. * M_PI);
    double bic = dNLL + 0.5 * log(N) * dP;
    double aic = dNLL + dP;

    result["GoodnessOfFit"] = NumericVector::create(_["NLE"]=dNLL,
                                                    _["LogDet"]=dLogDet,
                                                    _["Laplace"]=laplace,
                                                    _["BIC"]=bic,
                                                    _["AIC"]=aic);
    //group and fit results must be transposed
    result["Group"] = Z;

    List mix_list = mixture_output(data, W, Lambda, Err);
    result["Mixture"] = List::create(_["Weight"] = mix_list["Mixture"]);

    result["Fit"] = List::create(_["Estimate"]=mix_list["Estimate"],
                                 _["Upper"]=mix_list["Upper"],
                                 _["Lower"]=mix_list["Lower"]);

    return result;
}
*/