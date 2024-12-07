#ifndef MATRIX_H
#define MATRIX_H

void print_matrix(double **matrix,int n, int m);
double my_abs(double x);
double *diagonal(double **matrix,int n);
int *find_max_value(double **A,int n);


//creating_matrix
double **copy_matrix(double **A,int n);
double **init(int n, int m);
double **eye(int n);
double **zeros(int n);
double **diag(int n, double num);
double **generate_matrix_wiht_cond(int n,int m, int cond);
double **generate_matrix_with_eigvalues(int n, int m, double *eigvalues);

//matrix
double **mult_matrixes(double **matrix_1, int n_1, int  m_1, double ** matrix_2, int n_2, int m_2);
double **add_matrixes(double **matrix_1, int n_1, int  m_1, double ** matrix_2, int n_2, int m_2);
double **sub_matrixes(double **matrix_1, int n_1, int  m_1, double ** matrix_2, int n_2, int m_2);
double **add_num(double **matrix, int n, int m, double num);
double **sub_num(double **matrix, int n, int m, double num);
double **mult_num(double **matrix, int n, int m, double num);
double **transpose(double **matrix, int n, int m);
double matrix_norm_inf(double **matrix, int n, int m);
double trace(double **matrix, int n, int m);

//vectors
double vector_norm_2(double **vector, int n);
double scalar_product(double **vector_1, double **vector_2, int n);

//method QR for SLAU
double ***method_of_GH(double **matrix, int n, int m);
double **QR(double **A, int n, int m, double **b);

//method LDR for SLAU
double ***ldr(double **matrix, int n);
double sum(double **l, double **d, double **r, int i, int j, int k);
double **solution_with_downtriangle(double **l, double **b, int n);
double **solution_with_uppertriangle(double **l, double **b, int n);
double **solution_with_diag(double **l, double **b, int n);
double **solution_with_full_matrix(double **a, double **b, int n);

//leverie`s method for finding eigvalues
double *roots(double *coaf, double a, double b, int n);
double polinom(double *coaf, double x, int n);
double *get_traces(double **A,int n,int m);
double *get_coaf_of_hdet(double **A, int n, int m);
double *get_eigvalues(double **A, int n, int m);

//method QR for finding eigvaluew
double *GR_iegvalues(double **matrix,int n);


//Yakobi`s method for finding eigvalues
void jacobi_eigenvalues(double **A, int n);

//
double differentiation(double (*func)(double), double x);
double halfDivisionMethod(double a, double b, int *itr, int pogr, double (*func)(double));
double tangentMethod(double a, double b, int *itr, int pogr, double (*func)(double), double (*derivative)(double));
#endif
