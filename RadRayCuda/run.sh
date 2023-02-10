 rm out.txt
 nvcc main.cu radray.cu 3dmisc.cu -o main.exe
 nvprof ./main.exe
