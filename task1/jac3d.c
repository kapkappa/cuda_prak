/* Jacobi-3 program */

#include <math.h>
#include <stdio.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define L 384
#define ITMAX 100

int i, j, k, it;
double eps;
double MAXEPS = 0.5f;

int main(int an, char **as)
{
    //TODO: use malloc/new
    double A[L][L][L], B[L][L][L];
    double startt, endt;
    
    for (i = 0; i < L; i++)
        for (j = 0; j < L; j++)
            for (k = 0; k < L; k++)
            {
                A[i][j][k] = 0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                    B[i][j][k] = 0;
                else
                    B[i][j][k] = 4 + i + j + k;
            }    

    /* iteration loop */
    for (it = 1; it <= ITMAX; it++)
    {
        eps = 0;
        
        for (i = 1; i < L - 1; i++)
            for (j = 1; j < L - 1; j++)
                for (k = 1; k < L - 1; k++)
                {
                    double tmp = fabs(B[i][j][k] - A[i][j][k]);
                    eps = Max(tmp, eps);
                    A[i][j][k] = B[i][j][k];
                }

        for (i = 1; i < L - 1; i++)
            for (j = 1; j < L - 1; j++)
                for (k = 1; k < L - 1; k++)
                    B[i][j][k] = (A[i - 1][j][k] + A[i][j - 1][k] + A[i][j][k - 1] + A[i][j][k + 1] + A[i][j + 1][k] + A[i + 1][j][k]) / 6.0f;
        
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }

    printf(" Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", ITMAX);
    //TODO
    //printf(" Time in seconds =       %12.2lf\n", endt - startt);
    printf(" Operation type  =     floating point\n");
    //printf(" Verification    =       %12s\n", (fabs(eps - 5.058044) < 1e-11 ? "SUCCESSFUL" : "UNSUCCESSFUL"));

    printf(" END OF Jacobi3D Benchmark\n");
    return 0;
}
