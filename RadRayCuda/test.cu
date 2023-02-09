#include <stdio.h>
#include <stdlib.h>


struct porcozio {
    int *porcozii;
    char *ziofa;
    float *maremma;
};


__global__ void diosanto(porcozio *madimmite, int porconi) {

    printf("INIZIATO\n");
    //for (int i=0; i<porconi; i++) printf("%d %c %f\n", madimmite->porcozii[i], madimmite->ziofa[i], madimmite->maremma[i]);
    printf("FINITO\n");

}



int main() {
    
    int porconi = 5;

    porcozio *madimmite = (porcozio*) malloc(sizeof(porcozio));
    madimmite->porcozii = (int*) malloc(porconi*sizeof(int));
    madimmite->ziofa = (char*) malloc(porconi*sizeof(char));
    madimmite->maremma = (float*) malloc(porconi*sizeof(float));

    porcozio *dev_madimmite;

    cudaMalloc( (void**) &dev_madimmite, sizeof(porcozio));

    cudaMemcpy(dev_madimmite, madimmite, sizeof(porcozio), cudaMemcpyHostToDevice);
    diosanto<<<1,1>>>(dev_madimmite, porconi);


    return 0;
}