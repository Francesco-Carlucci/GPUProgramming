#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// define, if set to 1 it enables the comparison btwn parallel and serial version
#define COMPARE 1

// define used to select the point generation algorithm
#define POINT_GEN_PAR_RECT
//#define POINT_GEN_PAR
//#define POINT_GEN_SEQ

// define used to select the energies computation algorithm
#define ENEG_FULLY_PAR
//#define ENEG_PAR

#include "3dmisc.h"
#include "radray.h"

__constant__ ray dev_ray_traj;

__global__ void compute_energies(energy_point_s* dev_point_ens,int point_amt){  //,ray* dev_ray_traj

    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    point3d curr_ray_pos;
    double point_ray_dist;
    double bell_value;

    if(tid<point_amt) {
        curr_ray_pos = dev_ray_traj.start;
        for (int step = 0; step < N_STEPS; step++) {                                //Iterates over ray steps
            point_ray_dist = sqrt(pow(dev_point_ens[tid].pos.x - curr_ray_pos.x, 2) +
                                  pow(dev_point_ens[tid].pos.y - curr_ray_pos.y, 2) +
                                  pow(dev_point_ens[tid].pos.z - curr_ray_pos.z, 2));

            bell_value = ((1 / (2.506628274631 * 13.0)) * (double)exp(-0.5 * (double)pow((1 / 13.0 * (point_ray_dist/10 - 0)), 2)));
            dev_point_ens[tid].energy[step + 1] = dev_point_ens[tid].energy[step] +bell_value*
                                                   dev_ray_traj.energy_curve[step];

            curr_ray_pos.x += dev_ray_traj.delta.x;
            curr_ray_pos.y += dev_ray_traj.delta.y;
            curr_ray_pos.z += dev_ray_traj.delta.z;
        }
    }
}



__global__ void compute_energies_fully_parallel(energy_point_s* dev_point_ens,int point_amt){  //,ray* dev_ray_traj

    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int point_index=tid/N_STEPS;
    int en_index=tid%N_STEPS;
    //printf("\nthread n: %d %d %d\n",tid,point_index,en_index);

    point3d curr_ray_pos;
    double point_ray_dist;
    double bell_value;


    if(point_index<=point_amt) {
        curr_ray_pos = dev_ray_traj.start;
        curr_ray_pos.x += (dev_ray_traj.delta.x*(float) (en_index));
        curr_ray_pos.y += (dev_ray_traj.delta.y*(float) (en_index));
        curr_ray_pos.z += (dev_ray_traj.delta.z*(float) (en_index));

        point_ray_dist = sqrt(pow(dev_point_ens[point_index].pos.x - curr_ray_pos.x, 2) +
                                 pow(dev_point_ens[point_index].pos.y - curr_ray_pos.y, 2) +
                                 pow(dev_point_ens[point_index].pos.z - curr_ray_pos.z, 2));

        bell_value = ((1 / (2.506628274631 * 13.0)) * (double)exp(-0.5 * (double)pow((1 / 13.0 * (point_ray_dist/10 - 0)), 2)));

        dev_point_ens[point_index].energy[en_index+1] =bell_value*dev_ray_traj.energy_curve[en_index];
    }
}



int main() {  //pass file name and parameters through command line

    char* inpath = "../RadrayPy/out_all_points.txt";
    int cube_number; //i é l'indice del cubo
    point3d CUBE_GLOBAL_MAX = {0, 0, 0}, CUBE_GLOBAL_MIN = {1, 1, 1};
    cube cubes[MAX_CUBE];

    srand((unsigned int) time(0));

    cube_number=read_input(inpath,cubes,&CUBE_GLOBAL_MAX,&CUBE_GLOBAL_MIN);

    printf("-- Transient Mode analysis ... \n");

    point3d ray_start = {699, 1256, CUBE_GLOBAL_MAX.z};
    point3d ray_end = {699, 1256, CUBE_GLOBAL_MIN.z};
    ray ray_traj = fixed_ray(ray_start, ray_end, Constant);

    //ray* dev_ray_traj;
    energy_point * dev_point_ens;

    //cudaMalloc((void**) &dev_ray_traj,sizeof(ray));
    cudaError_t check=cudaMemcpyToSymbol(dev_ray_traj,&ray_traj,sizeof(ray));
    if(check!=cudaSuccess){
        printf("\nCuda memory error: %s, failed to fill ray trajectory\n",cudaGetErrorString(check));
    }


    FILE *fout_par = fopen("out_par.txt", "w");
    FILE *fout_seq = fopen("out_seq.txt", "w");

    //float ray_dist = distance(ray_traj.start, ray_traj.end);
    point3d curr_ray_pos;
    float point_ray_dist;
    point3d res = {10,10,10};
    //float dist_threshold = 10000;       ///:/=max_x_ray

    float cube_energy_sequential, cube_energy_parallel;

    // SECONDARY HIGH ENERGY RAYS GENERATION

    // data structures for secondary high energy rays
    ray ray_arr[N_RAYS];
    generate_rays(ray_arr, ray_traj, N_RAYS);
    // save rays to file
    fprintf(fout_par, "%d\n", N_RAYS);
    fprintf(fout_seq, "%d\n", N_RAYS);
    for (int i = 0; i < N_RAYS; i++) {
        fprintf(fout_par, "%f,%f,%f,%f,%f,%f\n", ray_arr[i].start.x, ray_arr[i].start.y, ray_arr[i].start.z, ray_arr[i].end.x, ray_arr[i].end.y, ray_arr[i].end.z);
        fprintf(fout_seq, "%f,%f,%f,%f,%f,%f\n", ray_arr[i].start.x, ray_arr[i].start.y, ray_arr[i].start.z, ray_arr[i].end.x, ray_arr[i].end.y, ray_arr[i].end.z);
    }


    // ################################################## SIMULATION PARALLEL ############################################


    printf("PARALLEL:\n");

#if COMPARE
    clock_t begin_par = clock();
#endif

    // for each cube in our system
    for(int cube_index = 0; cube_index < cube_number; cube_index++){

        // if the ray pass through the current cube we generate the points (atoms) and compute the energy
        if(cube_contains_ray(cubes[cube_index], ray_traj)){
            cube_energy_parallel = 0;
            printf("Raggio nel cubo %d - ", cube_index);

            // POINTS GENERATION ---------------

            clock_t begin_point_gen= clock();

#ifdef POINT_GEN_PAR_RECT 
            int x_amt, y_amt, z_amt;
            int offset=0;
            cubes[cube_index].rects[0].offset = 0;
            // for each triangle we compute the number of points and obtain the offset that its points will have in vector points
            for(int rect_index=1; rect_index<cubes[cube_index].rectN; rect_index++){
                point2d p1 = cubes[cube_index].rects[rect_index-1].p1;
                point2d p2 = cubes[cube_index].rects[rect_index-1].p2;
                x_amt = (p2.x - p1.x) / res.x;
                y_amt = (p2.y - p1.y) / res.y;
                z_amt = (cubes[cube_index].max.z - cubes[cube_index].min.z) / res.z;
                cubes[cube_index].rects[rect_index].offset = cubes[cube_index].rects[rect_index-1].offset + x_amt*y_amt*z_amt;
            }
            // we compute the total number of points used to allocate dev_points and points
            point2d p1 = cubes[cube_index].rects[cubes[cube_index].rectN-1].p1;
            point2d p2 = cubes[cube_index].rects[cubes[cube_index].rectN-1].p2;
            x_amt = (p2.x - p1.x) / res.x;
            y_amt = (p2.y - p1.y) / res.y;
            z_amt = (cubes[cube_index].max.z - cubes[cube_index].min.z) / res.z;
            cubes[cube_index].point_amt = cubes[cube_index].rects[cubes[cube_index].rectN-1].offset + x_amt*y_amt*z_amt;
            cubes[cube_index].points = (energy_point *) malloc(cubes[cube_index].point_amt * sizeof(energy_point));

            dev_point_ens = generate_points_in_rect_parallel(&(cubes[cube_index]), res);
#endif

#ifdef POINT_GEN_PAR
            generate_points_by_resolution_parallel(&(cubes[cube_index]), res, &dev_point_ens);
#endif

#ifdef POINT_GEN_SEQ
            generate_points_by_resolution(&cubes[cube_index], res);
            cudaMalloc((void**)&dev_point_ens,cubes[cube_index].point_amt *sizeof(energy_point));
            cudaMemcpy((void*) dev_point_ens, (void*) cubes[cube_index].points, cubes[cube_index].point_amt *sizeof(energy_point), cudaMemcpyHostToDevice);
#endif

            clock_t end_point_gen = clock();
            printf("Point gen: %f - ", (double) (end_point_gen - begin_point_gen) / CLOCKS_PER_SEC);

            // ENERGY COMPUTATION @@@@@@@@@@@@@@@@@@

            clock_t begin_eneg = clock();

#ifdef ENEG_PAR
            int nblocks = cubes[cube_index].point_amt/1024+1; 
            compute_energies<<<nblocks,1024>>>(dev_point_ens, dev_ray_traj, cubes[cube_index].point_amt);
            cudaMemcpy((void*) cubes[cube_index].points,(void*) dev_point_ens,cubes[cube_index].point_amt *sizeof(energy_point),cudaMemcpyDeviceToHost);
#endif

#ifdef ENEG_FULLY_PAR
            int nblocks=cubes[cube_index].point_amt*N_STEPS/1024+1;

            //compute_energies_fully_parallel<<<nblocks,1024>>>(dev_point_ens, dev_ray_traj,cubes[cube_index].point_amt);  // dev_ray_traj,
            compute_energies<<<nblocks,1024>>>(dev_point_ens,cubes[cube_index].point_amt);        //dev_ray_traj,
            cudaMemcpy((void*) cubes[cube_index].points,(void*) dev_point_ens,cubes[cube_index].point_amt *sizeof(energy_point),cudaMemcpyDeviceToHost);

            for(int point_index = 0; point_index < cubes[cube_index].point_amt; point_index++){
                /*for (int step=1; step <= N_STEPS; step++) {
                    cubes[cube_index].points[point_index].energy[step]+=cubes[cube_index].points[point_index].energy[step-1];
                }*/
                cube_energy_parallel += cubes[cube_index].points[point_index].energy[N_STEPS];
            }
#endif
            clock_t end_eneg = clock();
            printf("Eneg computation: %f - ", (double) (end_eneg - begin_eneg) / CLOCKS_PER_SEC);


            printf("Energy %f\n", cube_energy_parallel);
        }
        cudaFree(dev_point_ens);
    }
        
#if COMPARE
    clock_t end_par = clock();
#endif

    write_on_file(fout_par, cubes, cube_number, ray_traj);



    // ################################################## SIMULATION SEQUENTIAL ############################################



#if COMPARE

    // freeing and resetting data structures for time comparison
    for(int cube_index = 0; cube_index < cube_number; cube_index++)
        free(cubes[cube_index].points);

    printf("\nSEQUENTIAL:\n");

    clock_t begin_seq = clock();

    // for each cube in our system
    for(int cube_index = 0; cube_index < cube_number; cube_index++){

        // if the ray pass through the current cube we generate the points (atoms) and compute the energy
        if(cube_contains_ray(cubes[cube_index], ray_traj)){

            cube_energy_sequential = 0;
            printf("Raggio nel cubo %d - ", cube_index);

            clock_t begin_point_gen= clock();

#ifdef POINT_GEN_PAR_RECT
            //point_amt=somma su tutti i rettangoli, alloca preciso
            int x_amt = (cubes[cube_index].max.x - cubes[cube_index].min.x) / res.x;
            int y_amt = (cubes[cube_index].max.y - cubes[cube_index].min.y) / res.y;
            int z_amt = (cubes[cube_index].max.z - cubes[cube_index].min.z) / res.z;
            //cubes[cube_index].point_amt=x_amt * y_amt * z_amt;
            cubes[cube_index].points = (energy_point *) malloc(x_amt * y_amt * z_amt * sizeof(energy_point));
            int offset=0;
            for(int rect_index=0; rect_index<cubes[cube_index].rectN; rect_index++){
                offset=generate_points_in_rect(cubes[cube_index].rects[rect_index].p1, cubes[cube_index].rects[rect_index].p2, res, cubes[cube_index].points, cubes[cube_index].min.z, cubes[cube_index].max.z, offset);
            }
            cubes[cube_index].point_amt=offset;
#endif

#ifdef POINT_GEN_PAR
            generate_points_by_resolution(&cubes[cube_index], res);
#endif

#ifdef POINT_GEN_SEQ
            generate_points_by_resolution(&cubes[cube_index], res);
#endif

            clock_t end_point_gen = clock();
            printf("Point gen: %f - ", (double) (end_point_gen - begin_point_gen) / CLOCKS_PER_SEC);


            clock_t begin_eneg = clock();
            for(int point_index = 0; point_index < cubes[cube_index].point_amt; point_index++){
                    curr_ray_pos = ray_traj.start;
                    for(int step = 0; step < N_STEPS; step++){                                //Iterates over ray steps
                        point_ray_dist = distance(cubes[cube_index].points[point_index].pos, curr_ray_pos);
                        cubes[cube_index].points[point_index].energy[step + 1] =
                                cubes[cube_index].points[point_index].energy[step] +
                                bell(0, 13, point_ray_dist/10) *
                                ray_traj.energy_curve[step];  //1000 *

                        curr_ray_pos.x += ray_traj.delta.x;
                        curr_ray_pos.y += ray_traj.delta.y;
                        curr_ray_pos.z += ray_traj.delta.z;
                    }

                //printf("%f,%f,%f", cubes[cube_index].points[point_index].pos.x, cubes[cube_index].points[point_index].pos.y, cubes[cube_index].points[point_index].pos.z);
                cube_energy_sequential += cubes[cube_index].points[point_index].energy[N_STEPS];
            }
            clock_t end_eneg = clock();
            printf("Eneg computation: %f - ", (double) (end_eneg - begin_eneg) / CLOCKS_PER_SEC);

            printf("Energy %f\n", cube_energy_sequential);
        }
    }


    clock_t end_seq = clock();

    // writing on file sequential informations
    //write_on_file(fout_seq, cubes, cube_number, ray_traj);

    double sequential_time = (double) (end_seq - begin_seq) / CLOCKS_PER_SEC;
    double parallel_time =(double) (end_par - begin_par) / CLOCKS_PER_SEC;

    printf("\nParallelized computation: %f\n",parallel_time);
    printf("Sequential computation: %f\n",sequential_time);
    printf("Speedup= %.2f %% \n",(sequential_time/parallel_time)*100);

#endif

    // FREE DATA STRUCTURES AND CLOSE FILES

    cudaFree(dev_point_ens);
    //cudaFree(dev_ray_traj);   //Only if we don't use constant memory
    free_cubes(cubes, cube_number);
    fclose(fout_par);
    fclose(fout_seq);

    return 0;
}
