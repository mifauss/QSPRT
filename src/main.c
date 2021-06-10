#include <math.h>

#include <gsl/gsl_cdf.h>

#include "qsprt.h"


/* ----------- */
/* Define CDFs */
/* ----------- */


double
cdf_shift_in_mean(double t, int hyp)
{
    double mu = 0.0 + hyp;
    return gsl_cdf_ugaussian_P(t - mu);
}


double
cdf_shift_in_var(double t, int hyp)
{
    double sigma = sqrt(1.0 + hyp);
    return 2 * (gsl_cdf_gaussian_P(t, sigma) - gsl_cdf_gaussian_P(0, sigma));
}


/* ---------------- */
/* Example Problems */
/* ---------------- */


void
example_shift_in_mean()
{
    /* Domain of function rho */
    size_t NZ = 1001;
    double* Z = (double*) malloc(NZ * sizeof(double));
    qsprt_linspace(-10.0, 10.0, NZ, Z);
    
    /* Set of all possible quantizer thesholds */
    size_t NT = 51;
    double* T = (double*) malloc(NT * sizeof(double));
    qsprt_linspace(-2.5, 2.5, NT, T);
    
    /* Number of quantization levels */
    size_t K = 2;
    
    /* Allocate and initialize test */
    qsprt* test = qsprt_alloc(NZ, NT, gsl_interp_linear);
    qsprt_initialize(test, K, Z, T, &cdf_shift_in_mean);
    
    /* Set initial parameters */
    double kappa = 0.0;
    double l0 = 20.0;
    double l1 = 20.0;
    qsprt_set_parameters(test, kappa, l0, l1);
    
    /* Design optimal test that meets error probability constraints */
    double alpha = 0.01;        /* targeted alpha */
    double beta = 0.01;         /* targeted beta */
    double lambda_tol = 1e-3;   /* tolerance for calculating lambda */
    double rho_tol = 1e-6;      /* tolerance for calculating rho */
    size_t it_min = 20;         /* minimum number of iterations to calculate rho */
    bool verbose = true;        /* print progress */
    qsprt_design_constr(test, alpha, beta, lambda_tol, rho_tol, it_min, verbose);
    
    /* Evaluate optimal ASN */
    double asn_opt = qsprt_get_asn(test);
  
    /* Evaluate optimal quantizer threshold at log(z) = 0 */
    double t_opt[1];
    qsprt_get_t_opt(test, 0.0, t_opt);
    
    /* Design asymptotically optimal test that meets error probability constraints */
    qsprt_design_constr_asym(test, alpha, beta, lambda_tol, rho_tol, it_min, verbose);
    
    /* Evaluate asymptotically optimal ASN */
    double asn_asym = qsprt_get_asn(test);
    
     /* Evaluate asymptotically optimal quantizer threshold */
    double t_asym[1];
    qsprt_get_t_asym(test, t_asym);
    
    /* Evaluate ASN bounds */
    double asn_kl = qsprt_bound_asn_kl(test, alpha, beta);
    double asn_tv = qsprt_bound_asn_tv(test, alpha, beta);
    
    /* Print results */
    printf("\n");
    printf("Shift-in-mean test\n");
    printf("------------------\n");
    printf("ASN_opt  = %f\n", asn_opt);
    printf("t_opt    = %f\n", t_opt[0]);
    printf("ASN_asym = %f\n", asn_asym);
    printf("t_asym   = %f\n", t_asym[0]);
    printf("ASN_kl   = %f\n", asn_kl);
    printf("ASN_tv   = %f\n", asn_tv);
    printf("\n");
    
    /* Clean up */
    qsprt_free(test);
    free(Z);
    free(T);
}


void
example_shift_in_var()
{
    /* Domain of function rho */
    size_t NZ = 1001;
    double* Z = (double*) malloc(NZ * sizeof(double));
    qsprt_linspace(-10.0, 10.0, NZ, Z);
    
    /* Set of all possible quantizer thesholds */
    size_t NT = 61;
    double* T = (double*) malloc(NT * sizeof(double));
    qsprt_linspace(0.0, 3.0, NT, T);
    
    /* Number of quantization levels */
    size_t K = 2;
    
    /* Allocate and initialize test */
    qsprt* test = qsprt_alloc(NZ, NT, gsl_interp_linear);
    qsprt_initialize(test, K, Z, T, &cdf_shift_in_var);
    
    /* Set initial parameters */
    double PH1 = 0.035;
    double L = 1000.0;
    qsprt_set_parameters_bayes(test, PH1, L);
    
    /* Design optimal test */
    double rho_tol = 1e-6;      /* tolerance for calculating rho */
    size_t it_min = 20;         /* minimum number of iterations to calculate rho */
    bool verbose = false;       /* do not print progress */
    qsprt_design_unconstr(test, rho_tol, it_min, verbose);
    
    /* Evaluate optimal cost */
    double cost_opt = qsprt_get_cost(test);
    
    /* Evaluate optimal quantizer threshold at log(z) = 0 */
    double t_opt[1];
    qsprt_get_t_opt(test, 0.0, t_opt);
    
    /* Design asymptotically optimal test */
    qsprt_design_unconstr_asym(test, rho_tol, it_min, verbose);
    
    /* Evaluate asymptotically optimal cost */
    double cost_asym = qsprt_get_cost(test);
    
    /* Evaluate asymptotically optimal quantizer threshold */
    double t_asym[1];
    qsprt_get_t_asym(test, t_asym);
    
    /* Evaluate cost bounds */
    double ab_tol = 1e-6;   /* alpha-beta tolerance */
    double cost_bound = qsprt_bound_bayes(test, ab_tol, verbose);
    double cost_bound_ref = qsprt_bound_bayes_reference(test);
    
    /* Print results */
    printf("\n");
    printf("Shift-in-variance test\n");
    printf("----------------------\n");
    printf("cost_opt       = %f\n", cost_opt);
    printf("t_opt          = %f\n", t_opt[0]);
    printf("cost_asym      = %f\n", cost_asym);
    printf("t_asym         = %f\n", t_asym[0]);
    printf("cost_bound     = %f\n", cost_bound);
    printf("cost_bound_ref = %f\n", cost_bound_ref);
    printf("\n");
    
    /* Clean up */
    qsprt_free(test);
    free(Z);
    free(T);
}


/* ------------- */
/* Main Function */
/* ------------- */


int
main()
{   
    /* Modify the examples above to change parameter values */
  
    example_shift_in_mean();
  
    example_shift_in_var();
    
    return 0;
}


