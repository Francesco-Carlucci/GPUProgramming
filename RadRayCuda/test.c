#include <stdio.h>

int main() {
    int i = 0;
    int x = 52;
    int y = 30;
    int z = 3;

    while (i<x*y*z+3) {
        printf("%d %d %d\n", i%x, (i/x)%y, i/(x*y));
        i++; 
    }

    return 0;
}