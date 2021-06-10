#include <assert.h>
#include <stdbool.h>
#include <math.h>

#include <gsl/gsl_combination.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sort.h>

#include "qsprt.h"


/* ---------------- */
/* Helper Functions */
/* ---------------- */


void
qsprt_linspace(double x_min, double x_max, size_t N, double* x)
{   
    if (N == 1) {
        x[0] = x_min;
    } else {
        assert(x_max > x_min);
        double dx = (x_max - x_min) / (N - 1);
        for (size_t n = 0; n < N; n++) {
            x[n] = x_min + n * dx;
        }
    }
}


double
qsprt_eval_pmf(const qsprt* test, double *t, size_t x, size_t hyp)
{    
    if (x == 0) {
        return test->cdf(t[0], hyp);
    } else if (x == test->K-1) {
        return 1.0 - test->cdf(t[x-1], hyp);
    } else {
        return test->cdf(t[x], hyp) - test->cdf(t[x-1], hyp);
    }
}


double
qsprt_eval_pmf_vector(const qsprt* test, size_t* n, size_t x, size_t hyp)
{
    double *p = (hyp == 0) ? test->p0 : test->p1;

    if (x == 0) {
        return p[n[0]];
    } else if (x == test->K-1) {
        return 1.0 - p[n[x-1]];
    } else {
        return p[n[x]] - p[n[x-1]];
    }
}


/* ----------- */
/* Test Struct */
/* ----------- */


qsprt*
qsprt_alloc(size_t NZ, size_t NT, const gsl_interp_type* type)
{
    qsprt* test = (qsprt*)malloc(sizeof(qsprt));

    test->NZ = NZ;
    test->NT = NT;

    test->D = (double*) malloc(NZ * sizeof(double));

    test->p0 = (double*) malloc(NT * sizeof(double));
    test->p1 = (double*) malloc(NT * sizeof(double));

    test->d_spline = gsl_spline_alloc(type, NZ);
    test->acc = gsl_interp_accel_alloc();
    test->spline_type = type;
    
    test->status = ALLOCATED;

    return test;
}


void
qsprt_free(qsprt* test)
{
    free(test->D);
    free(test->p0);
    free(test->p1);

    gsl_spline_free(test->d_spline);
    gsl_interp_accel_free(test->acc);

    free(test);
}


void
qsprt_initialize(qsprt *test, size_t  K, double *Z, double *T, cdf_fp  cdf)
{
    test->Z = Z;
    test->T = T;

    for (size_t n = 0; n < test->NT; n++) {
        test->p0[n] = (*cdf)(test->T[n], 0);
        test->p1[n] = (*cdf)(test->T[n], 1);
    }

    assert(K > 1);
    test->K = K;
    test->cdf = cdf;
    
    test->kappa = 0.0;
    test->l0 = 0.0;
    test->l1 = 0.0;
    test->err0 = 0.0;
    test->err1 = 0.0;

    qsprt_reset_d_spline(test);
    
    test->status = INITIALIZED;
}


void
qsprt_set_parameters(qsprt* test, double kappa, double l0, double l1)
{
    assert(kappa <= 1.0);
    assert(kappa >= 0.0);
    assert(l0 > 0.0);
    assert(l1 > 0.0);
    
    test->kappa = kappa;
    test->l0 = l0;
    test->l1 = l1;
    
    test->status = SPECIFIED;
}


void
qsprt_set_parameters_bayes(qsprt* test, double PH1, double l)
{
    assert(PH1 <= 1.0);
    assert(PH1 >= 0.0);
    assert(l > 0.0);
    
    test->kappa = PH1;
    test->l0 = (1.0-PH1) * l;
    test->l1 = PH1 * l;
    
    test->status = SPECIFIED;
}


/* ---------------- */
/* Spline Functions */
/* ---------------- */


void
qsprt_reset_d_spline(qsprt* test)
{
    for (size_t n = 0; n < test->NZ; n++) {
        test->D[n] = qsprt_g_func(test, test->Z[n]);
    }

    gsl_spline_init(test->d_spline, test->Z, test->D, test->NZ);
    gsl_interp_accel_reset(test->acc);
}


void
qsprt_update_d_spline(qsprt* test, double* d)
{
    for (size_t n = 0; n < test->NZ; n++) {
        test->D[n] = d[n];
    }

    gsl_spline_init(test->d_spline, test->Z, test->D, test->NZ);
    gsl_interp_accel_reset(test->acc);
}


double
qsprt_iterate_d_spline(qsprt* test, double* d)
{
    double diff;

    #pragma omp parallel for
    for (size_t n = 0; n < test->NZ; n++) {
        d[n] = qsprt_get_min_cost(test, test->Z[n], &qsprt_rho_cost, NULL);
    }

    diff = qsprt_rho_func(test, 0.0);
    qsprt_update_d_spline(test, d);
    diff -= qsprt_rho_func(test, 0.0);

    return diff;
}


int
qsprt_check_spline(const qsprt* test)
{
    double kappa = test->kappa;
    bool failure = false;
    
    double z_lo = test->Z[0];
    double g_lo = qsprt_g_func(test, z_lo);
    double r_lo = (1.0 - kappa) + kappa * exp(z_lo);
    
    double z_up = test->Z[test->NZ-1];
    double g_up = qsprt_g_func(test, z_up);
    double r_up = (1.0 - kappa) + kappa * exp(z_up);
    
    if (r_lo + test->D[0] < g_lo) {
        printf("   Lower threshold out of bounds.\n");
        failure = true;
    }

    if (r_up + test->D[test->NZ - 1] < g_up) {
        printf("   Upper threshold out of bounds.\n");
        failure = true;
    }

    return failure ? GSL_FAILURE : GSL_SUCCESS;
}


/* ------------ */
/* Similarities */
/* ------------ */


double
qsprt_kld0_func(const qsprt* test, double z)
{
    return -z;
}


double
qsprt_kld1_func(const qsprt* test, double z)
{
    return z*exp(z);
}


double
qsprt_g_func(const qsprt* test, double z)
{
    double l0 = test->l0;
    double l1 = test->l1;
    return GSL_MIN_DBL(l0, l1 * exp(z));
}


double
qsprt_d_func(const qsprt* test, double z)
{
    return gsl_spline_eval(test->d_spline, z, test->acc);
}


double
qsprt_rho_func(const qsprt* test, double z)
{
    double g_val = qsprt_g_func(test, z);

    if (z < test->Z[0] || z > test->Z[test->NZ - 1]) {
        return g_val;
    } else {
        double kappa = test->kappa;
        double d_val = qsprt_d_func(test, z);
        double r_val = (1-kappa) + kappa * exp(z);
        return GSL_MIN_DBL(g_val, r_val + d_val);
    }
}


double
qsprt_eval_similarity(const qsprt *test, double z, size_t *n, sim_fp sim_func)
{
    double z_next, p0, p1;
    double sim = 0.0;
    
    for (size_t x = 0; x < test->K; x++) {
        p0 = qsprt_eval_pmf_vector(test, n, x, 0);
        p1 = qsprt_eval_pmf_vector(test, n, x, 1);

        assert(p0 >= 0.0);
        assert(p1 >= 0.0);
        
        if (p0 > 0.0) {
            z_next = z + log(p1 / p0);
            sim += sim_func(test, z_next) * p0;
        }
    }
    
    return sim;
}


/* -------------- */
/* Cost Functions */
/* -------------- */


double
qsprt_neg_kld0_cost(const qsprt *test, double z, size_t *n)
{  
    return -qsprt_eval_similarity(test, z, n, &qsprt_kld0_func);
}


double
qsprt_neg_kld1_cost(const qsprt *test, double z, size_t *n)
{    
    return -qsprt_eval_similarity(test, z, n, &qsprt_kld1_func);  
}


double
qsprt_asym_cost(const qsprt *test, double z, size_t *n)
{
    double kappa = test->kappa;
    double D0 = qsprt_eval_similarity(test, z, n, &qsprt_kld0_func);
    double D1 = qsprt_eval_similarity(test, z, n, &qsprt_kld1_func);  
    
    return (1-kappa)/D0 + kappa/D1;
}


double
qsprt_g_cost(const qsprt *test, double z, size_t *n)
{ 
       return qsprt_eval_similarity(test, z, n, &qsprt_g_func);;
}


double
qsprt_rho_cost(const qsprt *test, double z, size_t *n)
{
    return qsprt_eval_similarity(test, z, n, &qsprt_rho_func);
}


double
qsprt_get_min_cost(const qsprt *test, double z, cost_fp cost_func, double *t)
{
    gsl_combination* c = gsl_combination_calloc(test->NT, test->K-1);
    size_t* n = gsl_combination_data(c);
    double min_cost = GSL_POSINF;
    int status;

    do {       
        double cost = (*cost_func)(test, z, n);
        if (cost < min_cost) {
            min_cost = cost;
            if (t != NULL) {
                for (size_t i = 0; i < test->K-1; i++) {
                    t[i] = test->T[n[i]];
                }
            }
        }
        status = gsl_combination_next(c);
    } while (status == GSL_SUCCESS);

    gsl_combination_free(c);

    return min_cost;
}


/* ------------------------- */
/* Unconstrained Test Design */
/* ------------------------- */


int
qsprt_design_unconstr(qsprt *test, double tol, int itmin, bool verbose)
{    
    if (test->status < SPECIFIED) {
        printf("Parameters need to be specified before designing a test.\n");
        return GSL_FAILURE;
    }
    
    double* d = (double*) malloc(test->NZ * sizeof(double));
    double diff;
    int iter = 0;
    
    qsprt_reset_d_spline(test);
    
    do {
        iter++;
        diff = qsprt_iterate_d_spline(test, d);
        if (verbose) {
            printf("%5d: delta = %e\n", iter, diff);
        }
    } while (fabs(diff) > tol || iter <= itmin);

    free(d);
    
    if (qsprt_check_spline(test) == GSL_SUCCESS) {
        test->status = SOLVED_UNCONSTR;
        return GSL_SUCCESS;
    } else {
        return GSL_FAILURE;
    }
}


int
qsprt_design_unconstr_asym(qsprt *test, double tol, int itmin, bool verbose)
{   
    if (test->status < SPECIFIED) {
        printf("Parameters need to be specified before designing a test.\n");
        return GSL_FAILURE;
    }
    
    /* Get optimal thresholds... */
    double t[test->K-1];
    qsprt_get_min_cost(test, 0.0, &qsprt_asym_cost, t);
    
    /* Set up new test */
    qsprt* test_asym = qsprt_alloc(test->NZ, test->K-1, test->spline_type);
    qsprt_initialize(test_asym, test->K, test->Z, t, test->cdf);
    qsprt_set_parameters(test_asym, test->kappa, test->l0, test->l1);
    
    /* Solve */
    int result = qsprt_design_unconstr(test_asym, tol, itmin, verbose);
    
    qsprt_update_d_spline(test, test_asym->D);
    test->status = test_asym->status;
    
    qsprt_free(test_asym);

    return result;
}



/* ----------------------- */
/* Constrained Test Design */
/* ----------------------- */


double
qsprt_dual_cost(const gsl_vector* l, void* void_param)
{   
    dual_param* param = (dual_param*) void_param;
    qsprt* test = param->test;
    
    test->l0 = gsl_vector_get(l, 0);
    test->l1 = gsl_vector_get(l, 1);

    qsprt_design_unconstr(param->test, param->tol, param->itmin, false);

    double rho0 = qsprt_rho_func(param->test, 0.0);

    return -(rho0 - param->err0*test->l0 - param->err1*test->l1);
}


double
qsprt_dual_cost_bayes(const gsl_vector* l, void* void_param)
{   
    dual_param_bayes* param = (dual_param_bayes*) void_param;
    qsprt* test = param->test;
    
    double PH1 = test->kappa;
    test->l0 = (1.0-PH1)*gsl_vector_get(l, 0);
    test->l1 =      PH1 *gsl_vector_get(l, 0);

    qsprt_design_unconstr(param->test, param->tol, param->itmin, false);

    double rho0 = qsprt_rho_func(param->test, 0.0);

    return -(rho0 - param->err*(test->l0 + test->l1));
}


int
qsprt_design_constr(qsprt  *test,
                    double  alpha,
                    double  beta,
                    double  l_tol,
                    double  d_tol,
                    int     d_itmin,
                    bool    verbose)
{    
    if (test->status < SPECIFIED) {
        printf("Parameters need to be specified before designing a test.\n");
        return GSL_FAILURE;
    }
    
    const gsl_multimin_fminimizer_type* alg = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer* solver = gsl_multimin_fminimizer_alloc(alg, 2);
    dual_param param = { test, alpha, beta, d_tol, d_itmin };

    gsl_multimin_function objective;
    objective.n = 2;
    objective.f = &qsprt_dual_cost;
    objective.params = (void*) &param;

    gsl_vector* step_size = gsl_vector_alloc(2);
    gsl_vector_set(step_size, 0, 0.1 * test->l0);
    gsl_vector_set(step_size, 1, 0.1 * test->l1);

    gsl_vector* l = gsl_vector_alloc(2);
    gsl_vector_set(l, 0, test->l0);
    gsl_vector_set(l, 1, test->l1);

    gsl_multimin_fminimizer_set(solver, &objective, l, step_size);

    int iter = 0;
    int status;
    double simplex_size;
    
    if (verbose) {
        printf("\nFinding optimal cost coefficients...\n");
    }

    do {
        iter++;
        status = gsl_multimin_fminimizer_iterate(solver);

        if (status) break;

        simplex_size = gsl_multimin_fminimizer_size(solver);
        status = gsl_multimin_test_size(simplex_size, l_tol);

        if (verbose) {
            printf("%5d: lambda = (%6.4f, %6.4f), asn = %10.3e, simplex size = %9.3e\n",
                   iter,
                   gsl_vector_get(solver->x, 0),
                   gsl_vector_get(solver->x, 1),
                   -solver->fval,
                   simplex_size);
        }
    } while (status == GSL_CONTINUE);

    test->l0 = gsl_vector_get(solver->x, 0);
    test->l1 = gsl_vector_get(solver->x, 1);
    
    gsl_multimin_fminimizer_free(solver);
    gsl_vector_free(step_size);
    gsl_vector_free(l);
    
    if (status == GSL_SUCCESS) {
        status = qsprt_design_unconstr(test, d_tol, d_itmin, false);
        if (status == GSL_SUCCESS) {
            test->status = SOLVED_CONSTR;
            test->err0 = alpha;
            test->err1 = beta;
        }
    }
    
    return status;
}


int
qsprt_design_constr_bayes(qsprt  *test,
                          double  err,
                          double  l_tol,
                          double  d_tol,
                          int     d_itmin,
                          bool    verbose)
{    
    if (test->status < SPECIFIED) {
        printf("Parameters need to be specified before designing a test.\n");
        return GSL_FAILURE;
    }
    
    const gsl_multimin_fminimizer_type* alg = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer* solver = gsl_multimin_fminimizer_alloc(alg, 1);
    dual_param param = { test, err, d_tol, d_itmin };

    gsl_multimin_function dual_func;
    dual_func.n = 1;
    dual_func.f = &qsprt_dual_cost_bayes;
    dual_func.params = (void*) &param;

    gsl_vector* step_size = gsl_vector_alloc(1);
    gsl_vector_set(step_size, 0, 0.1 * test->l1 / test->kappa);

    gsl_vector* l = gsl_vector_alloc(1);
    gsl_vector_set(l, 0, test->l1 / test->kappa);

    gsl_multimin_fminimizer_set(solver, &dual_func, l, step_size);

    int iter = 0;
    int status;
    double simplex_size;
    
    if (verbose) {
        printf("\nFinding optimal Bayes cost...\n");
    }
    
    do {
        iter++;
        status = gsl_multimin_fminimizer_iterate(solver);

        if (status) break;

        simplex_size = gsl_multimin_fminimizer_size(solver);
        status = gsl_multimin_test_size(simplex_size, l_tol);

        if (verbose) {
            printf("%5d: lambda = %6.4f, asn = %10.3e, simplex size = %9.3e\n",
                   iter,
                   gsl_vector_get(solver->x, 0),
                   -solver->fval,
                   simplex_size);
        }
    } while (status == GSL_CONTINUE);

    double PH1 = test->kappa;
    test->l0 = (1.0-PH1) * gsl_vector_get(solver->x, 0);
    test->l1 =      PH1  * gsl_vector_get(solver->x, 0);
    
    gsl_multimin_fminimizer_free(solver);
    gsl_vector_free(step_size);
    gsl_vector_free(l);
    
    if (status == GSL_SUCCESS) {
        status = qsprt_design_unconstr(test, d_tol, d_itmin, false);
        if (status == GSL_SUCCESS) {
            test->status = SOLVED_CONSTR;
            test->err0 = err;
            test->err1 = err;
        }
    }
    
    return status;
}


int
qsprt_design_constr_asym(qsprt  *test,
                         double  alpha,
                         double  beta,
                         double  l_tol,
                         double  d_tol,
                         int     d_itmin,
                         bool    verbose)
{
    if (test->status < SPECIFIED) {
        printf("Parameters need to be specified before designing a test.\n");
        return GSL_FAILURE;
    }
    
    /* Get optimal thresholds... */
    double t[test->K-1];
    qsprt_get_min_cost(test, 0.0, &qsprt_asym_cost, (double*) t);
    
    /* Set up new test */
    qsprt* test_asym = qsprt_alloc(test->NZ, test->K-1, gsl_interp_linear);
    qsprt_initialize(test_asym, test->K, test->Z, t, test->cdf);
    qsprt_set_parameters(test_asym, test->kappa, test->l0, test->l1);

    /* Solve */
    int result = qsprt_design_constr(test_asym, alpha, beta, l_tol, d_tol, d_itmin, verbose);
    
    test->l0 = test_asym->l0;
    test->l1 = test_asym->l1;
    
    qsprt_update_d_spline(test, test_asym->D);
    qsprt_free(test_asym);
    
    if (result == GSL_SUCCESS) {
        test->status = SOLVED_CONSTR;
        test->err0 = alpha;
        test->err1 = beta;
    }

    return result;
}


int
qsprt_design_constr_asym_bayes(qsprt  *test,
                               double  err,
                               double  l_tol,
                               double  d_tol,
                               int     d_itmin,
                               bool    verbose)
{    
    if (test->status < SPECIFIED) {
        printf("Parameters need to be specified before designing a test.\n");
        return GSL_FAILURE;
    }
    
    /* Get optimal thresholds */
    double t[test->K-1];
    qsprt_get_min_cost(test, 0.0, &qsprt_asym_cost, t);

    /* Set up new test */
    qsprt* test_asym = qsprt_alloc(test->NZ, test->K-1, gsl_interp_linear);
    qsprt_initialize(test_asym, test->K, test->Z, t, test->cdf);
    qsprt_set_parameters(test_asym, test->kappa, test->l0, test->l1);

    /* Solve */
    int result = qsprt_design_constr_bayes(test_asym, err, l_tol, d_tol, d_itmin, verbose);
    
    test->l0 = test_asym->l0;
    test->l1 = test_asym->l1;
    
    qsprt_update_d_spline(test, test_asym->D);
    qsprt_free(test_asym);
    
    if (result == GSL_SUCCESS) {
        test->status = SOLVED_CONSTR;
        test->err0 = err;
        test->err1 = err;
    }

    return result;
}


/* -------------- */
/* Access Results */
/* -------------- */


double
qsprt_get_cost(const qsprt* test)
{
    if (test->status < SOLVED_UNCONSTR) {
        printf("A test needs to be designed before evaluating its cost.\n");
        return GSL_NAN;
    }
    
    return qsprt_rho_func(test, 0.0);
}


double
qsprt_get_asn(const qsprt* test)
{
    if (test->status < SOLVED_CONSTR) {
        printf("A constrained test needs to be designed before evaluating its ASN.\n");
        return GSL_NAN;
    }
    
    return qsprt_rho_func(test, 0.0) - test->err0*test->l0 - test->err1*test->l1;
}


void
qsprt_get_t_opt(const qsprt* test, double z, double* t)
{
    if (test->status < SOLVED_UNCONSTR) {
        printf("A test needs to be designed before evaluating its thresholds.\n");
    }
    qsprt_get_min_cost(test, z, &qsprt_rho_cost, t);
}


void
qsprt_get_t_asym(const qsprt* test, double* t)
{
    if (test->status < SPECIFIED) {
        printf("Test parameters need to be specified before evaluating the asymptotic thresholds.\n");
    }
    qsprt_get_min_cost(test, 0.0, &qsprt_asym_cost, t);
}


/* ------ */
/* Bounds */
/* ------ */


double
qsprt_bound_bayes_reference(qsprt* test)
{   
    if (test->status < SPECIFIED) {
        printf("Parameters need to be specified before evaluating the bound.\n");
        return GSL_NAN;
    }
    
    double PH1 = test->kappa;
    
    double D[2];
    D[0] = -qsprt_get_min_cost(test, 0.0, &qsprt_neg_kld0_cost, NULL);
    D[1] = -qsprt_get_min_cost(test, 0.0, &qsprt_neg_kld1_cost, NULL);
  
    double Dmin = GSL_MIN_DBL(D[0], D[1]);
    
    double L = test->l1/PH1;
    double a = GSL_MAX_DBL(1.0, 2.0/Dmin);
    double b = (1.0/3.0)*(Dmin - 1.0/a);
    double Lstar = GSL_MAX_DBL( GSL_MAX_DBL(2.0*a, 4.0/b), 1.0/pow(b, 2) );
    double K1 = (1 + a*b + log(Lstar)) / Dmin;
    
    double p[2] = {  1.0-PH1, PH1 };
    double w[2] = { (1.0-PH1)/PH1, PH1/(1.0-PH1) };
    
    double bound = 0.0;
    for (size_t i=0; i<2; i++){
        bound += p[i] * (log(L-1.0) - log(w[i])) / D[i];
    }
    bound -= K1;
        
    return GSL_MAX_DBL(bound, 0.0);
}


double 
qsprt_bound_asn_kl(qsprt *test, double alpha, double beta)
{   
    if (test->status < SPECIFIED) {
        printf("Parameters need to be specified before evaluating the bound.\n");
        return GSL_NAN;
    }
    
    double kappa = test->kappa;
    
    double D0 = -qsprt_get_min_cost(test, 0.0, &qsprt_neg_kld0_cost, NULL);
    double D1 = -qsprt_get_min_cost(test, 0.0, &qsprt_neg_kld1_cost, NULL);
    
    double N0 = alpha*log(alpha/(1-beta )) + (1-alpha)*log((1-alpha)/beta );
    double N1 = beta *log(beta /(1-alpha)) + (1-beta )*log((1-beta )/alpha);
    
    return (1-kappa)*N0/D0 + kappa*N1/D1;
}


double 
qsprt_bound_asn_tv(qsprt *test, double alpha, double beta)
{   
    double l0 = test->l0;
    double l1 = test->l1;
    
    test->l0 = 1.0;
    test->l1 = 1.0;
    
    double DTV = 1.0 - qsprt_get_min_cost(test, 0.0, &qsprt_g_cost, NULL);
    
    test->l0 = l0;
    test->l1 = l1;
    
    return (1.0-alpha-beta)/DTV;
}


double
qsprt_bound_bayes_cost(const gsl_vector* x, void* void_param)
{   
    bayes_bound_param* param = (bayes_bound_param*) void_param;
    
    double kappa = param->kappa;
    double l0 = param->l0;
    double l1 = param->l1;
    double D0 = param->D0;
    double D1 = param->D1;
    
    double a = 1.0/(1 + exp(gsl_vector_get(x, 0)));
    double b = 1.0/(1 + exp(gsl_vector_get(x, 1)));

    double N0 = a*log(a/(1-b)) + (1-a)*log((1-a)/b);
    double N1 = b*log(b/(1-a)) + (1-b)*log((1-b)/a);

    return (1-kappa)*N0/D0 + kappa*N1/D1 + a*l0 + b*l1; 
}


double
qsprt_bound_bayes(qsprt  *test,
                  double  err_tol,
                  bool    verbose)
{    
    if (test->status < SPECIFIED) {
        printf("Parameters need to be specified before evaluating a bound.\n");
        return GSL_FAILURE;
    }
    
    double D0 = -qsprt_get_min_cost(test, 0.0, &qsprt_neg_kld0_cost, NULL);
    double D1 = -qsprt_get_min_cost(test, 0.0, &qsprt_neg_kld1_cost, NULL);
    
    const gsl_multimin_fminimizer_type* alg = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer* solver = gsl_multimin_fminimizer_alloc(alg, 2);
    bayes_bound_param param = { test->kappa, test->l0, test->l1, D0, D1 };

    gsl_multimin_function cost_func;
    cost_func.n = 2;
    cost_func.f = &qsprt_bound_bayes_cost;
    cost_func.params = (void*) &param;

    gsl_vector* step_size = gsl_vector_alloc(2);
    gsl_vector_set_all(step_size, 1.0);

    gsl_vector* x = gsl_vector_alloc(2);
    gsl_vector_set_all(x, 1.0);

    gsl_multimin_fminimizer_set(solver, &cost_func, x, step_size);

    int iter = 0;
    int status;
    double simplex_size;

    do {
        iter++;
        status = gsl_multimin_fminimizer_iterate(solver);

        if (status) break;

        simplex_size = gsl_multimin_fminimizer_size(solver);
        status = gsl_multimin_test_size(simplex_size, err_tol);

        if (verbose) {
            printf("%5d: (a, b) = (%6.4f, %6.4f), cost = %10.3e, simplex size = %9.3e\n",
                   iter,
                   gsl_vector_get(solver->x, 0),
                   gsl_vector_get(solver->x, 1),
                   solver->fval,
                   simplex_size);
        }
    } while (status == GSL_CONTINUE);
    
    double bound = qsprt_bound_bayes_cost(solver->x, (void*) &param);
    
    gsl_multimin_fminimizer_free(solver);
    gsl_vector_free(step_size);
    gsl_vector_free(x);
    
    if (status != GSL_SUCCESS) {
        printf("Warning: algorithm terminated early, result might be inaccurate!\n");
    }
    
    return bound;
}

