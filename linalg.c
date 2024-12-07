#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#define EXP 2.7182818284590452354
#define PI 3.14159265358979323846


void print_matrix(double **matrix,int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (i == n - 1 && j == m - 1) {
                printf("%lf", matrix[i][j]);
            } else if (j == m - 1) {
                printf("%lf\n", matrix[i][j]);
            } else {
                printf("%lf ", matrix[i][j]);
            }
        }
    }
}
double *diagonal(double **matrix,int n) {
    double *arr = malloc(sizeof(double)*n);
    for(int i = 0; i < n; i++) {
        arr[i] = matrix[i][i];
    }
    return arr;
}
double my_abs(double x) {
    int sign = x > 0 ? 1 : -1;
    return x*sign;
}
int *find_max_value(double **A,int n) {
    double max = -10000;
    int *indexes = malloc(sizeof(int)*2);
    for(int j =0; j < n; j++) {
        for(int i = 0; i < n; i++) {
            if(j != i) {
                if(max == -10000) {
                    max = A[i][j];
                    indexes[0] = i;
                    indexes[1] = j;
                } else if(my_abs(max) < my_abs(A[i][j])) {
                    max = A[i][j];
                    indexes[0] = i;
                    indexes[1] = j;
                }
            }
        }
    }
    return indexes;
}

//init
double **init(int n, int m) {
    double **matrix = malloc(n*m*sizeof(double) + n*sizeof(double*));
    double *start = (double*)matrix + n;
    for (int i = 0; i < n; i++) {
        matrix[i] = start + i*m;
    }
    return matrix;
}

//matrix
double **mult_matrixes(double **matrix_1, int n_1, int  m_1, double ** matrix_2, int n_2, int m_2) {
    double **buff = NULL;
    if(m_1 == n_2) {
        buff = init(n_1,m_2);
        for (int i = 0; i < n_1; i++) {
            for (int j = 0; j < m_2; j++) {
                double sum = 0;
                for(int k = 0; k < m_1; k++) {
                    sum += matrix_1[i][k] * matrix_2[k][j];
                }
                buff[i][j] = sum;
            }
        }
    }
    return buff;
}
double **add_matrixes(double **matrix_1, int n_1, int  m_1, double ** matrix_2, int n_2, int m_2) {
    double **buff = NULL;
    if(n_1 == n_2 && m_1 == m_2) {
        for (int i = 0; i < n_1; i++) {
            for (int j = 0; j < m_1; j++) {
                matrix_1[i][j] = matrix_1[i][j] + matrix_2[i][j];
            }
        }
        buff = matrix_1;
    }
    return buff;
}
double **sub_matrixes(double **matrix_1, int n_1, int  m_1, double ** matrix_2, int n_2, int m_2) {
    double **buff = NULL;
    if(n_1 == n_2 && m_1 == m_2) {
        for (int i = 0; i < n_1; i++) {
            for (int j = 0; j < m_1; j++) {
                matrix_1[i][j] = matrix_1[i][j] - matrix_2[i][j];
            }
        }
        buff = matrix_1;
    }
    return buff;
}
double **add_num(double **matrix, int n, int m, double num) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            matrix[i][j] = matrix[i][j] + num;
        }
    }
    return matrix;
}
double **sub_num(double **matrix, int n, int m, double num) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            matrix[i][j] = matrix[i][j] - num;
        }
    }
    return matrix;
}
double **mult_num(double **matrix, int n, int m, double num) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            matrix[i][j] = matrix[i][j] * num;
        }
    }
    return matrix;
}
double **transpose(double **matrix, int n, int m) {
    double **res = init(m,n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            res[j][i] = matrix[i][j];
        }
    }
    return res;
}
double matrix_norm_inf(double **matrix, int n, int m) {
    double res = 0;
    for(int i = 0; i < n; i++) {
        double sum = 0;
        for(int j = 0; j < m; j++) {
            int sign = matrix[i][j] >= 0 ? 1 : 0;
            sum += sign * matrix[i][j];
        }
        res = sum > res? sum : res;
    }
    return res;
}
double trace(double **matrix, int n, int m) {
    double trace = 0;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
	    if(i == j) {	
            	trace += matrix[i][j];
	    }
        }
    }
    return trace;
}

//vectors
double vector_norm_2(double **vector, int n) {
    double res = 0;
    for(int i =0; i < n; i++) {
        res += pow(vector[i][0],2);
    }
    return pow(res,0.5);
}
double scalar_product(double **vector_1, double **vector_2, int n) {
    double res = 0;
    for(int i = 0; i < n; i++) {
        res += vector_1[i][0] * vector_2[i][0];
    }
    return res;
}

//creating_matrix
double **copy_matrix(double **A,int n) {
    double **copy = init(n,n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            copy[i][j] = A[i][j];
        }
    }
    return copy;
}
double **eye(int n) {
    double **A = init(n,n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(j == i) {
                A[i][j] = 1;
            } else {
                A[i][j] = 0;
            }
        }
    }
    return A;
}
double **zeros(int n) {
    double **matrix = malloc(n*n*sizeof(double) + n*sizeof(double*));
    double *start = (double*)(matrix + n);
    for(int i = 0; i < n; i++){
        matrix[i] = start + i*n;
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j< n; j++) {
            matrix[i][j] = 0;
        }
    }
    return matrix;
}
double **diag(int n, double num) {
    double **matrix = malloc(n*n*sizeof(double) + n*sizeof(double*));
    double *start = (double*)(matrix + n);
    for(int i = 0; i < n; i++){
        matrix[i] = start + i*n;
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j< n; j++) {
            if(i == j) {
                matrix[i][j] = num;
            } else {
                matrix[i][j] = 0;
            }
        }
    }
    return matrix;    
}
double **generate_matrix_with_cond(int n,int m, int cond) {
    double **M = init(n,m);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(i == j) {
                M[i][j] = i*(cond - 1)/(n-1) + 1;
            } else if(j > i){
                M[i][j] = 1;
            } else {
                M[i][j] = 0;
            }
        }
    }


    double **W = init(n,1);
    srand(time(NULL));
    for(int i = 0; i < n; i++) {
        double r = rand()%1000;
        W[i][0] = r/1000;
    }
    double **W_t = transpose(W,n,1);
    
    double **E = init(n,n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(i == j) {
                E[i][j] = 1;
            } else {
                E[i][j] = 0;
            }
        }
    }

    double coaf = 2/(vector_norm_2(W,n)*vector_norm_2(W,n));
    double **mult = mult_matrixes(W,n,1,W_t,1,n);
    mult = mult_num(mult,n,n,coaf);

    E = sub_matrixes(E,n,n,mult,n,n);

    double **temp_res = mult_matrixes(E,n,n,M,n,n);
    double **E_t = transpose(E,n,n);

    double **temp_res_2 = mult_matrixes(temp_res,n,n,E_t,n,n);


    free(M);
    free(E);
    free(E_t);
    free(temp_res);
    free(W);
    free(W_t);
    free(mult);
    return temp_res_2;
}
double **generate_matrix_with_eigvalues(int n, int m, double *eigvalues) {
    double **M = init(n,m);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(i == j) {
                M[i][j] = eigvalues[i];
            } else if(j > i){
                M[i][j] = 0;
            } else {
                M[i][j] = 0;
            }
            
        }
    }



    double **W = init(n,1);
    srand(time(NULL));
    for(int i = 0; i < n; i++) {
        double r = rand()%1000;
        W[i][0] = r/1000;
    }
    double **W_t = transpose(W,n,1);
    
    double **E = init(n,n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(i == j) {
                E[i][j] = 1;
            } else {
                E[i][j] = 0;
            }
        }
    }

    double coaf = 2/(vector_norm_2(W,n)*vector_norm_2(W,n));
    double **mult = mult_matrixes(W,n,1,W_t,1,n);
    mult = mult_num(mult,n,n,coaf);

    E = sub_matrixes(E,n,n,mult,n,n);

    double **temp_res = mult_matrixes(E,n,n,M,n,n);
    double **E_t = transpose(E,n,n);

    double **temp_res_2 = mult_matrixes(temp_res,n,n,E_t,n,n);


    free(M);
    free(E);
    free(E_t);
    free(temp_res);
    free(W);
    free(W_t);
    free(mult);
    return temp_res_2;    
}

//method QR for SLAU
double ***method_of_GH(double **matrix, int n, int m) {
    double **ONB = init(n,m);
    double **RA = init(n,m);
    for(int i = 0; i < n; i++) {
        double **u = init(n,1); 
        for(int j = 0; j < n; j++) {
            u[j][0] = matrix[j][i];
        }
        for(int j = 0; j < i; j++) {
            double **a = init(n,1);
            double **q = init(n,1);
            for(int k = 0; k < n; k++) {
                a[k][0] = matrix[k][i];
                q[k][0] = ONB[k][j];
            }
            double scalar = scalar_product(a,q,n);
            u = sub_matrixes(u,n,1,mult_num(q,n,1,scalar),n,1);
            free(a);
            free(q);
        }

        RA[i][i] = 1 / vector_norm_2(u,n);
        for(int k = 0; k < i; k++) {
            double p = 0;
            for(int j = k; j < i; j++) {
                double **a = init(n,1);
                double **q = init(n,1);
                for(int g = 0; g < n; g++) {
                    a[g][0] = matrix[g][i];
                    q[g][0] = ONB[g][j];
                }
                p += scalar_product(a,q,n)*RA[j][k];
                free(a);
                free(q);
            }
            p *= -RA[i][i];
            RA[i][k] = p;
        }

        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                if(j > i) {
                    RA[i][j] = 0;
                }
            }
        }
        for(int j = 0; j < n; j++) {
            ONB[j][i] = u[j][0] / vector_norm_2(u,n);
        }
        free(u);
    }

    double ***res = malloc(3*sizeof(double**));
    res[0] = ONB;
    res[1] = RA;
    return res;
}
double **QR(double **A, int n, int m, double **b) {
    double ***buff = method_of_GH(A,n,m);
    double **Q_t = transpose(buff[0],n,m);
    double **R_inv = transpose(buff[1],n,m);
    double **temp = mult_matrixes(R_inv,n,n,Q_t,n,n);
    double **res = mult_matrixes(temp,n,n,b,n,1);
    free(buff[0]);
    free(buff[1]);
    free(buff);
    free(Q_t);
    free(R_inv);
    free(temp);
    return res;
}

//method LDR for SLAU
double sum(double **l, double **d, double **r, int i, int j, int k) {
    double sum = 0;
    for(int n = 0; n < k; n++) {
        sum += l[i][n]*d[n][n]*r[n][j];
    } 
    return sum;
}
double ***ldr(double **matrix, int n) {
    double **l = diag(n, 1);
    double **d = zeros(n);
    double **r = diag(n,1);
    for(int m = 0; m < n; m++) {
        d[m][m] = matrix[m][m] - sum(l,d,r,m,m,m);
        for(int j = m+1; j < n; j++) {
            l[j][m] = (matrix[j][m] - sum(l,d,r,j,m,m)) / d[m][m];
            r[m][j] = (matrix[m][j] - sum(l,d,r,m,j,m)) / d[m][m];
        }
    }
    double ***res = malloc(3*sizeof(double**));
    res[0] = l;
    res[1] = d;
    res[2] = r;
    return res;
}
double **solution_with_downtriangle(double **l, double **b, int n) {
    double **x = init(n,1);
    for(int i = 0; i < n; i++) {
        double temp = 0;
        for(int j = 0; j < i; j++) {
            temp += l[i][j]*x[j][0];
        }
        x[i][0] = (b[i][0] - temp) / l[i][i];
    }
    return x;
}
double **solution_with_uppertriangle(double **l, double **b, int n) {
    double **x = init(n,1);
    for(int i = n-1; i > -1; i--) {
        double temp = 0;
        for(int j = i+1; j < n; j++) {
            temp += l[i][j]*x[j][0];
        }
        x[i][0] = (b[i][0] - temp) / l[i][i];
    }
    return x;
}
double **solution_with_diag(double **l, double **b, int n) {
    double **x = init(n,1);
    for(int i = 0; i < n; i++) {
        double temp = 0;
        x[i][0] = b[i][0] / l[i][i];
    }
    return x;
}
double **solution_with_full_matrix(double **a, double **b, int n) {
    double ***arr = ldr(a,n);
    double **z = solution_with_downtriangle(arr[0],b,n);
    double **y = solution_with_diag(arr[1], z,n);
    double **x = solution_with_uppertriangle(arr[2], y, n);
    free(z);
    free(y);
    return x;
}


//leverie`s method for finding eigvalues
double polinom(double *coaf, double x, int n) {
    double res = 0;
    for(int i = n-1; i >= 0; i--) {
        res += pow(x,i)*coaf[n-i-1];
    }
    return res;
}
double *roots(double *coaf, double a, double b, int n) {
    int counter = 0;
    double left = a, right;
    double *roots = malloc(sizeof(double) * (n)), eps = pow(EXP,-20);
    for(double i = a; i <= b; i += 0.001) {
        right = i;
        if(polinom(coaf,left,n) * polinom(coaf,right,n) < 0) {
            while (right - left > 2*eps) {
                double c = (left + right) / 2;
                if (polinom(coaf,left, n) * polinom(coaf,c, n) < 0) {
                    right = c;
                } else {
                    left = c;
                }
            }
            roots[counter] = (right+left)/2;
            counter++;
            left =right;
        }
        
    }
    return roots;
}
double *get_traces(double **A,int n,int m) {
    double **prev = init(n,n);
    for(int i =0; i<n;i++) {
        for(int j = 0; j < n; j++) {
            prev[i][j] = A[i][j];
        }
    }
    double *s = malloc(sizeof(double) * (n+1));
    s[1] = trace(A,n,n);
    for(int i = 2; i < n+1; i++) {
        double **current = mult_matrixes(prev,n,n,A,n,n);
        s[i] = trace(current,n,n);
        free(prev);
        prev = current;
    }
    free(prev);
    return s;
}
double *get_coaf_of_hdet(double **A, int n, int m) {
    double *s = get_traces(A,n,m);
    double *p = malloc(sizeof(double)*(n+1));
    p[0] = 1;
    for(int k = 1; k < n+1; k++) {
        p[k] = s[k];
        for(int i = 1; i < k; i++) {
            p[k] += p[i]*s[k-i];
        }
        p[k] = -p[k]/k;
    }
    free(s);
    return p;
}
double *get_eigvalues(double **A, int n, int m) {
    double *p = get_coaf_of_hdet(A,n,m);
    double *eigvalues = roots(p,-matrix_norm_inf(A,n,n)-20, matrix_norm_inf(A,n,n)+20, n+1); 
    free(p);
    return eigvalues;
}

//method QR for finding eigvaluew
double *GR_iegvalues(double **matrix,int n) {
    double **A_k = copy_matrix(matrix,n);
    double eps = pow(EXP,-17);
    int i = 0;
    for(i = 0; i < 100; i++) {
        double ***GR = method_of_GH(A_k,n,n);
        double **Q_t = transpose(GR[0],n,n);
        double **R = mult_matrixes(Q_t,n,n,A_k,n,n);
        double **new_A_k = mult_matrixes(R,n,n,GR[0],n,n);
        double *diag1 = diagonal(A_k,n);
        double *diag2 = diagonal(new_A_k,n);
        double diff = 0;
        for(int j = 0; j < n;j++) {
            diff += my_abs(diag1[j]-diag2[j]);
        }
        free(A_k);
        A_k = new_A_k;
        free(diag1);
        free(diag2);
        free(R);
        free(Q_t);
        free(GR[0]);
        free(GR[1]);
        free(GR);
        if(diff < eps) {
            break;
        }
    }

    double *eigvalues = malloc(sizeof(double)*n);
    for(int i = 0; i < n;i++) {
        eigvalues[i] = A_k[i][i];
    }
    free(A_k);
    printf("count of iterations = %d\n\n",i);
    return eigvalues;
}

//Yakobi`s method for finding eigvalues
void jacobi_eigenvalues(double **matrix, int n) {
    double **A = copy_matrix(matrix,n);
    double **eigvectors = eye(n);
    double e = pow(EXP,-15);
    int iter = 0;
    for(; iter < 10000; iter++) {
        int *indexes = find_max_value(A,n);
        int p = indexes[0];
        int q = indexes[1];
        if(my_abs(A[p][q]) < e) {
            free(indexes);
            break;
        } 
        double fi;
        if(A[p][p] == A[q][q]) {
            fi = PI / 4;
        } else {
            fi = 0.5 * atan(2 * A[p][q] / (A[q][q] - A[p][p]));
        }
        double cos_fi = cos(fi);
        double sin_fi = sin(fi);
        double **R = eye(n);
        R[p][p] = cos_fi;
        R[q][q] = cos_fi;
        R[p][q] = sin_fi;
        R[q][p] = -sin_fi;
        double **R_transpose = transpose(R,n,n);
        double **temp_1 = mult_matrixes(R_transpose,n,n,A,n,n);
        double **temp_2 = mult_matrixes(temp_1,n,n,R,n,n);
        free(A);
        A = temp_2;
        double **temp_3 = mult_matrixes(eigvectors,n,n,R,n,n);
        free(eigvectors);
        eigvectors = temp_3;
        free(R);
        free(R_transpose);
        free(indexes);
        free(temp_1);
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(i == j) {
                printf("eig%d = %lf\n",i+1,A[i][j]);
            }
        }
    }
    printf("\n\n");
    print_matrix(eigvectors,n,n);
    printf("\n\n");

    printf("iter = %d",iter);

    free(A);
    free(eigvectors);
}
//

double differentiation(double (*func)(double), double x) {
    return (func(x + 0.00001) - func(x)) / 0.00001;
}
double halfDivisionMethod(double a, double b, int *itr, int pogr, double (*func)(double)) {
    double res;
    double eps = 1 * pow(2.718, -pogr);
    (*itr) = 0;
    while (b - a > 2*eps) {
        (*itr)++;
        double c = (b + a)/2;
        if(func(c) == 0) {
            res = c;
        } else if (func(a)*func(c) < 0) {
            b = c;
        } else {
            a = c;
        }
    }
    res = (b + a)/ 2;
    return res;
}
double tangentMethod(double a, double b, int *itr, int pogr, double (*func)(double), double (*derivative)(double)) {
    double x1, x2, m1, M2;
    double eps = 1 * pow(2.718, -pogr);
    m1 = my_abs(differentiation(func,a)) < my_abs(differentiation(func,b)) ? my_abs(differentiation(func,a)) : my_abs(differentiation(func,b)); 
    M2 = my_abs(differentiation(derivative,a)) < my_abs(differentiation(derivative,b)) ? my_abs(differentiation(derivative,a)) : my_abs(differentiation(derivative,b)); 
    (*itr) = 0;
    if(func(a) * differentiation(derivative, a) > 0) {
        x1 = a;
    } else {
        x1 = b;
    }
    x2 = x1 - func(x1)/derivative(x1);
    (*itr)++;
    while (my_abs(x2 - x1) > pow(2*eps*m1/M2, 0.5)) {
        x1 = x2;
        x2 = x1 - func(x1)/derivative(x1);
        (*itr)++;
    }
    return x2;
}














