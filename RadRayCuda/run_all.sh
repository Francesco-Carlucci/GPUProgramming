rm out.txt
rm out_all_points.txt
py gds.py $1
nvcc main.cu radray.cu 3dmisc.cu -o main.exe
nvprof ./main.exe
py read.py