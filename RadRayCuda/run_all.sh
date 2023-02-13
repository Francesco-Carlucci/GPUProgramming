rm out.txt
rm out_all_points.txt
python3 gds.py $1
nvcc main.cu radray.cu 3dmisc.cu -o main.exe
./main.exe
python3 read.py $1