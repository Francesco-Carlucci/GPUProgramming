
build: main.cu radray.cu 3dmisc.cu
	nvcc main.cu radray.cu 3dmisc.cu -o main.exe

run_nobuid: main.exe
	./main.exe

run: main.cu radray.cu 3dmisc.cu
	nvcc main.cu radray.cu 3dmisc.cu -o main.exe
	./main.exe

profile: main.exe
	nvprof ./main.exe

build_profile: main.cu radray.cu 3dmisc.cu main.exe
	nvcc main.cu radray.cu 3dmisc.cu -o main.exe
	nvprof ./main.exe
	nvprof --print-gpu-trace ./main.exe