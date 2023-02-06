#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define COMPARE 1

#include "3dmisc.h"
#include "radray.h"


__global__ void compute_energies(energy_point_s* dev_point_ens,ray* dev_ray_traj,int point_amt){

    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    point3d curr_ray_pos;
    double point_ray_dist;
    double bell_value;
    //energy_point_s* point;
    //point=&(dev_point_ens[tid]);
    if(tid<point_amt) {
        curr_ray_pos = dev_ray_traj->start;
        for (int step = 0; step < N_STEPS; step++) {                                //Iterates over ray steps
            point_ray_dist = sqrt(pow(dev_point_ens[tid].pos.x - curr_ray_pos.x, 2) +
                                  pow(dev_point_ens[tid].pos.y - curr_ray_pos.y, 2) +
                                  pow(dev_point_ens[tid].pos.z - curr_ray_pos.z, 2));

            bell_value = ((1 / (2.506628274631 * 130.0)) * (double)exp(-0.5 * (double)pow((1 / 130.0 * (point_ray_dist/10 - 0)), 2)));
            /*if(tid==0 && step==0){  //to be activated for debugging purposes
                printf("distance: %f bell: %lf\n",point_ray_dist,bell_value);
            }*/
            dev_point_ens[tid].energy[step + 1] = dev_point_ens[tid].energy[step] +bell_value*
                                                   dev_ray_traj->energy_curve[step];

            curr_ray_pos.x += dev_ray_traj->delta.x;
            curr_ray_pos.y += dev_ray_traj->delta.y;
            curr_ray_pos.z += dev_ray_traj->delta.z;
        }
    }
}



__global__ void compute_energies_fully_parallel(energy_point_s* dev_point_ens,ray* dev_ray_traj,int point_amt){

    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int point_index=tid/N_STEPS;
    int en_index=tid%N_STEPS;
    //printf("\nthread n: %d %d %d\n",tid,point_index,en_index);

    point3d curr_ray_pos;
    double point_ray_dist;
    double bell_value;


    if(point_index<=point_amt) {
        curr_ray_pos = dev_ray_traj->start;
        curr_ray_pos.x += (dev_ray_traj->delta.x*(float) (en_index));
        curr_ray_pos.y += (dev_ray_traj->delta.y*(float) (en_index));
        curr_ray_pos.z += (dev_ray_traj->delta.z*(float) (en_index));

        point_ray_dist = sqrt(pow(dev_point_ens[point_index].pos.x - curr_ray_pos.x, 2) +
                                 pow(dev_point_ens[point_index].pos.y - curr_ray_pos.y, 2) +
                                 pow(dev_point_ens[point_index].pos.z - curr_ray_pos.z, 2));

        bell_value = ((1 / (2.506628274631 * 130.0)) * (double)exp(-0.5 * (double)pow((1 / 130.0 * (point_ray_dist/10 - 0)), 2)));

        dev_point_ens[point_index].energy[en_index+1] =bell_value*dev_ray_traj->energy_curve[en_index];
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

    ray* dev_ray_traj;
    energy_point * dev_point_ens;

    cudaMalloc((void**) &dev_ray_traj,sizeof(ray));
    cudaMemcpy(dev_ray_traj,&ray_traj,sizeof(ray),cudaMemcpyHostToDevice);

    FILE *fout = fopen("out.txt", "w");

    //float ray_dist = distance(ray_traj.start, ray_traj.end);
    point3d curr_ray_pos;
    float point_ray_dist;
    point3d res = {30, 30, 30};
    float dist_threshold = 10000;       ///:/=max_x_ray

    float cube_energy;

    // SECONDARY HIGH ENERGY RAYS GENERATION

    // data structures for secondary high energy rays
    ray ray_arr[N_RAYS];
    generate_rays(ray_arr, ray_traj, N_RAYS);
    // save rays to file
    fprintf(fout, "%d\n", N_RAYS);
    for (int i = 0; i < N_RAYS; i++) {
        fprintf(fout, "%f,%f,%f,%f,%f,%f\n", ray_arr[i].start.x, ray_arr[i].start.y, ray_arr[i].start.z, ray_arr[i].end.x, ray_arr[i].end.y, ray_arr[i].end.z);
    }

    // SIMULATION

    // for each cube in our system
    for(int cube_index = 0; cube_index < cube_number; cube_index++){               //Iterates over the cubes

        // if the ray pass through the current cube we generate the points (atoms) and compute the energy
        if(cube_contains_ray(cubes[cube_index], ray_traj)){                     //Check if the ray is in the cube
            cube_energy = 0;
            printf("Raggio nel cubo %d - ", cube_index);
            fprintf(fout, "%d\n", cube_index);

            // POINTS GENERATION
            //energy_point *dev_point_ens;
            generate_points_by_resolution_parallel(&cubes[cube_index], res,&dev_point_ens);
            //cudaMemcpy((void*) cubes[cube_index].points,(void*) dev_point_ens,cubes[cube_index].point_amt *sizeof(energy_point),cudaMemcpyDeviceToHost);
            //cube currcube=cubes[cube_index];
            //generate_points_by_resolution(&currcube,res);
#if COMPARE
            clock_t begin = clock();
#endif
            int nblocks=cubes[cube_index].point_amt*N_STEPS/1024+1;   //*N_STEPS
            /***
             * migliorare gestione blocchi per evitare la warp divergence di point_amt
             */
            compute_energies_fully_parallel<<<nblocks,1024>>>(dev_point_ens, dev_ray_traj, cubes[cube_index].point_amt);
            cudaMemcpy((void*) cubes[cube_index].points,(void*) dev_point_ens,cubes[cube_index].point_amt *sizeof(energy_point),cudaMemcpyDeviceToHost);

            for(int point_index = 0; point_index < cubes[cube_index].point_amt; point_index++){
                fprintf(fout, "%f,%f,%f", cubes[cube_index].points[point_index].pos.x, cubes[cube_index].points[point_index].pos.y, cubes[cube_index].points[point_index].pos.z);
                for (int step=1; step <= N_STEPS; step++) {
                    cubes[cube_index].points[point_index].energy[step]+=cubes[cube_index].points[point_index].energy[step-1];
                    fprintf(fout, ",%f", cubes[cube_index].points[point_index].energy[step]);
                }
                fprintf(fout, "\n");
                if(!(cubes[cube_index].points[point_index].pos.x==0 && cubes[cube_index].points[point_index].pos.y==0 &&cubes[cube_index].points[point_index].pos.z==0)) {
                    cube_energy += cubes[cube_index].points[point_index].energy[N_STEPS];
                }
            }
#if COMPARE
            clock_t end = clock();
            double parallel_time=(double) (end - begin)/ CLOCKS_PER_SEC;
            printf("\nParallelized computation: %f\n",parallel_time);
            printf("Energia: %f\n", cube_energy);
            cube_energy=0;
            begin=clock();
            free(cubes[cube_index].points);

            generate_points_by_resolution(&cubes[cube_index],res);

            for(int point_index = 0; point_index < cubes[cube_index].point_amt; point_index++){
                    curr_ray_pos = ray_traj.start;
                    for(int step = 0; step < N_STEPS; step++){                                //Iterates over ray steps
                        point_ray_dist = distance(cubes[cube_index].points[point_index].pos, curr_ray_pos);
                        cubes[cube_index].points[point_index].energy[step + 1] =
                                cubes[cube_index].points[point_index].energy[step] +
                                bell(0, 130, point_ray_dist/10) *
                                ray_traj.energy_curve[step];  //1000 *

                        curr_ray_pos.x += ray_traj.delta.x;
                        curr_ray_pos.y += ray_traj.delta.y;
                        curr_ray_pos.z += ray_traj.delta.z;
                    }

                fprintf(fout, "%f,%f,%f", cubes[cube_index].points[point_index].pos.x, cubes[cube_index].points[point_index].pos.y, cubes[cube_index].points[point_index].pos.z);
                for (int step=1; step <= N_STEPS; step++) fprintf(fout, ",%f", cubes[cube_index].points[point_index].energy[step]);
                fprintf(fout, "\n");
                cube_energy += cubes[cube_index].points[point_index].energy[N_STEPS];
            }
            end = clock();
            double sequential_time=(double) (end - begin)/ CLOCKS_PER_SEC;
            printf("\nsequential computation: %f\n",sequential_time);
            printf("speedup= %.2f %% \n",(1-parallel_time/sequential_time)*100);
#endif
            printf("Energia: %f\n\n", cube_energy);
        }
    }

    // FREE DATA STRUCTURES AND CLOSE FILES

    cudaFree(dev_point_ens);
    cudaFree(dev_ray_traj);
    free_cubes(cubes, cube_number);
    //fclose(fin);
    fclose(fout);

    return 0;
}
