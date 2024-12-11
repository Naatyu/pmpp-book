#include <stdio.h>
#include <stdlib.h>

// Compute vector sum C_h = A_h + B_h
void vec_add(float *A_h, float *B_h, float *C_h, int n)
{
    for (int i = 0; i < n; i++)
    {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main()
{
    // Declare variables
    int size = 16;
    float *A_h = malloc(size * sizeof(float));
    float *B_h = malloc(size * sizeof(float));
    float *C_h = malloc(size * sizeof(float));

    // Init of vectors
    for (int i = 0; i < size; i++)
    {
        A_h[i] = i + 1.25f;
        B_h[i] = i * 1.25f;
    }

    // Perform vector addition
    vec_add(A_h, B_h, C_h, size);

    // Print results
    for (int i = 0; i < size; i++)
    {
        printf("elem %d is %.2f\n", i, C_h[i]);
    }

    // Free memory !!
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}