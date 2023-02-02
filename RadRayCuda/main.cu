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
        for (int step = 0; step < 100; step++) {                                //Iterates over ray steps
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
    float dist_threshold = 10000;       //=max_x_ray

    float cube_energy;

    // data structures for secondary high energy rays
    ray ray_arr[N_RAYS];
    generate_rays(ray_arr, ray_traj, N_RAYS);

    fprintf(fout, "%d\n", N_RAYS);
    // save rays to file
    for (int i = 0; i < N_RAYS; i++) {
        fprintf(fout, "%f,%f,%f,%f,%f,%f\n", ray_arr[i].start.x, ray_arr[i].start.y, ray_arr[i].start.z, ray_arr[i].end.x, ray_arr[i].end.y, ray_arr[i].end.z);
    }
    for(int cube_index = 0; cube_index < cube_number; cube_index++){               //Iterates over the cubes
        if(cube_contains_ray(cubes[cube_index], ray_traj)){                     //Check if the ray is in the cube
            cube_energy = 0;
            printf("Raggio in cubo %d - ", cube_index);
            fprintf(fout, "%d\n", cube_index);
            //generate_points_by_amount(&cubes[cube_index], MAX_POINTS);
            generate_points_by_resolution(&cubes[cube_index], res);
            //for(int point_index = 0; point_index < cubes[cube_index].point_amt; point_index++){  //Iterates over points in the cube
#if COMPARE
            clock_t begin = clock();
#endif
            int nblocks=cubes[cube_index].point_amt/1024+1;
            cudaMalloc((void**)&dev_point_ens,cubes[cube_index].point_amt *sizeof(energy_point));   //(N_STEPS + 1)*sizeof(float)
            //initialize points array on device, first and last element to 0
            cudaMemcpy((void*) dev_point_ens,(void*) cubes[cube_index].points,
                       cubes[cube_index].point_amt *sizeof(energy_point),cudaMemcpyHostToDevice);
            /***
             * migliorare gestione blocchi per evitare la warp divergence di point_amt
             */
            compute_energies<<<nblocks,1024>>>(dev_point_ens,dev_ray_traj,cubes[cube_index].point_amt);

            cudaMemcpy((void*) cubes[cube_index].points,(void*) dev_point_ens,cubes[cube_index].point_amt *sizeof(energy_point),cudaMemcpyDeviceToHost); //((void**) &dev_point_ens[i],(N_STEPS + 1)*sizeof(float));

            for(int point_index = 0; point_index < cubes[cube_index].point_amt; point_index++){
                fprintf(fout, "%f,%f,%f", cubes[cube_index].points[point_index].pos.x, cubes[cube_index].points[point_index].pos.y, cubes[cube_index].points[point_index].pos.z);
                for (int step=1; step <= N_STEPS; step++) fprintf(fout, ",%f", cubes[cube_index].points[point_index].energy[step]);
                fprintf(fout, "\n");
                cube_energy += cubes[cube_index].points[point_index].energy[N_STEPS];
            }
#if COMPARE
            clock_t end = clock();
            printf("\nparallelized computation: %f\n",(double)(end - begin) / CLOCKS_PER_SEC);
            printf("Energia: %f\n", cube_energy);
            cube_energy=0;
            begin=clock();

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
            printf("\nsequential computation: %f\n",(double)(end - begin) / CLOCKS_PER_SEC);
#endif
            printf("Energia: %f\n\n", cube_energy);
        }
    }
    free_cubes(cubes, cube_number);
    //fclose(fin);
    fclose(fout);
    return 0;

    return 0;
}