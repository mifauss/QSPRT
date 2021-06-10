#pragma once

#include <stdbool.h> 

#include <gsl/gsl_spline.h>
#include <gsl/gsl_vector.h>


/* ----- */
/* Types */
/* ----- */


enum test_status 
{
    ALLOCATED = 0, 
    INITIALIZED = 1, 
    SPECIFIED = 2, 
    SOLVED_UNCONSTR = 3,
    SOLVED_CONSTR = 4
};


typedef struct
{
    size_t NZ, NT, K;
    double kappa, l0, l1, err0, err1;
    double *Z, *T, *D, *p0, *p1;
    gsl_spline* d_spline;
    gsl_interp_accel* acc;
    const gsl_interp_type* spline_type;
    double (*cdf)(double, int);
    enum test_status status;
} qsprt;


typedef struct
{
    qsprt* test;
    double err0, err1, tol;
    int itmin;
} dual_param;


typedef struct
{
    qsprt* test;
    double err, tol;
    int itmin;
} dual_param_bayes;


typedef struct
{
    double kappa, l0, l1, D0, D1;
} bayes_bound_param;


typedef double (*cdf_fp)(double, int);


typedef double (*sim_fp)(const qsprt*, double);


typedef double (*cost_fp)(const qsprt*, double, size_t*);


/* ---------------- */
/* Helper Functions */
/* ---------------- */


void
qsprt_linspace(double x_min, double x_max, size_t N, double* x);


double
qsprt_eval_pmf(const qsprt *test, double *t, size_t x, size_t hyp);


double
qsprt_eval_pmf_vector(const qsprt *test, size_t *n, size_t x, size_t hyp);


/* ----------- */
/* Test Struct */
/* ----------- */


qsprt*
qsprt_alloc(size_t NZ, size_t NT, const gsl_interp_type* type);


void
qsprt_free(qsprt* test);


void
qsprt_initialize(qsprt *test, size_t K, double *Z, double *T, cdf_fp cdf);


void
qsprt_set_parameters(qsprt* test, double kappa, double l0, double l1);


void
qsprt_set_parameters_bayes(qsprt* test, double PH1, double l);


/* ---------------- */
/* Spline Functions */
/* ---------------- */


void
qsprt_reset_d_spline(qsprt* test);


void
qsprt_update_d_spline(qsprt* test, double* d);


double
qsprt_iterate_d_spline(qsprt* test, double* d);


/* ------------ */
/* Similarities */
/* ------------ */


double
qsprt_kld0_func(const qsprt* test, double z);


double
qsprt_kld1_func(const qsprt* test, double z);


double
qsprt_g_func(const qsprt* test, double z);


double
qsprt_d_func(const qsprt* test, double z);


double
qsprt_rho_func(const qsprt* test, double z);


double
qsprt_eval_similarity(const qsprt *test, double z, size_t *n, sim_fp sim_func);


/* -------------- */
/* Cost Functions */
/* -------------- */


double
qsprt_neg_kld0_cost(const qsprt *test, double z, size_t *n);


double
qsprt_neg_kld1_cost(const qsprt *test, double z, size_t *n);


double
qsprt_asym_cost(const qsprt *test, double z, size_t *n);


double
qsprt_g_cost(const qsprt *test, double z, size_t *n);


double
qsprt_rho_cost(const qsprt *test, double z, size_t *n);


double
qsprt_get_min_cost(const qsprt *test, double z, cost_fp cost_func, double *t);


/* ------------------------- */
/* Unconstrained Test Design */
/* ------------------------- */


int
qsprt_design_unconstr(qsprt *test, double tol, int itmin, bool verbose);


int
qsprt_design_unconstr_asym(qsprt *test, double  tol, int itmin, bool verbose);


/* ----------------------- */
/* Constrained Test Design */
/* ----------------------- */


double
qsprt_dual_cost(const gsl_vector* L, void* void_param);


double
qsprt_dual_cost_bayes(const gsl_vector* L, void* void_param);


int
qsprt_design_constr(qsprt  *test,
                    double  err0,
                    double  err1,
                    double  l_tol,
                    double  d_tol,
                    int     d_itmin,
                    bool    verbose);


int
qsprt_design_constr_bayes(qsprt  *test,
                          double  err,
                          double  l_tol,
                          double  d_tol,
                          int     d_itmin,
                          bool    verbose);


int
qsprt_design_constr_asym(qsprt  *test,
                         double  err0,
                         double  err1,
                         double  lambda_tol,
                         double  d_tol,
                         int     d_itmin,
                         bool    verbose);


int
qsprt_design_constr_asym_bayes(qsprt  *test,
                               double  err,
                               double  l_tol,
                               double  d_tol,
                               int     d_itmin,
                               bool    verbose);





/* -------------- */
/* Access Results */
/* -------------- */


double
qsprt_get_cost(const qsprt* test);


double
qsprt_get_asn(const qsprt* test);


void
qsprt_get_t_opt(const qsprt* test, double z, double* t);


void
qsprt_get_t_asym(const qsprt* test, double* t);


/* ------ */
/* Bounds */
/* ------ */


double
qsprt_bound_bayes_reference(qsprt* test);


double 
qsprt_bound_asn_kl(qsprt *test, double a, double b);


double 
qsprt_bound_asn_tv(qsprt *test, double a, double b);


double
qsprt_bound_bayes(qsprt  *test,
                  double  err_tol,
                  bool    verbose);

