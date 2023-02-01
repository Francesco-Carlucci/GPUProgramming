#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "3dmisc.h"
#include "radray.h"

int read_input(char* inpath,cube cubes[],point3d* CUBE_GLOBAL_MAX, point3d* CUBE_GLOBAL_MIN){
    FILE* fin;
    cube t;
    int cube_number=0,i;

    //Open the file
    fin = fopen(inpath, "r");
    if(fin==NULL){
        printf("Unable to read the file!");
        exit(1);
    }
    /***
     * Read file, format for each boundary:                 \n
     * N    layer                                           \n
     * minz maxz                                            \n
     * x    y   (repeated N times)                          \n
     * Calculates minx and miny of the cube Bounding Box    \n
     * Calculates Global bounding box                       \n
     */
    while(fscanf(fin, "%d %d", &t.N, &t.layer_n) != EOF) {
        t.limits = (point2d *) malloc(t.N * sizeof(point2d));
        t.limits = (point2d *) malloc(t.N * sizeof(point2d));
        fscanf(fin, "%f %f",&t.min.z,&t.max.z);
        for (i = 0; i < t.N; i++) {
            fscanf(fin, "%f %f", &(t.limits[i].x), &t.limits[i].y);
            if (i == 0) {
                t.min.x = t.limits[0].x;
                t.min.y = t.limits[0].y;
                t.max.x = t.limits[0].x;
                t.max.y = t.limits[0].y;
            } else {
                t.max.x = t.limits[i].x > t.max.x ? t.limits[i].x : t.max.x;
                t.min.x = t.limits[i].x < t.min.x ? t.limits[i].x : t.min.x;
                t.max.y = t.limits[i].y > t.max.y ? t.limits[i].y : t.max.y;
                t.min.y = t.limits[i].y < t.min.y ? t.limits[i].y : t.min.y;
            }
        }
        if (t.max.x > CUBE_GLOBAL_MAX->x) { CUBE_GLOBAL_MAX->x = t.max.x; }  // computes global max and min, in which the ray must pass
        if (t.max.y > CUBE_GLOBAL_MAX->y) { CUBE_GLOBAL_MAX->y = t.max.y; }
        if (t.max.z > CUBE_GLOBAL_MAX->z) { CUBE_GLOBAL_MAX->z = t.max.z; }
        if (t.min.x < CUBE_GLOBAL_MIN->x) { CUBE_GLOBAL_MIN->x = t.min.x; }
        if (t.min.y < CUBE_GLOBAL_MIN->y) { CUBE_GLOBAL_MIN->y = t.min.y; }
        if (t.min.z < CUBE_GLOBAL_MIN->z) { CUBE_GLOBAL_MIN->z = t.min.z; }
        t.points=NULL;
        cubes[cube_number] = t;
        cube_number++;
    }
    fclose(fin);
    return cube_number;
}

int main(int argc, char* argv[]){
    char* inpath = "../RadrayPy/out_all_points.txt";
    int cube_number; //i é l'indice del cubo
    point3d CUBE_GLOBAL_MAX = {0, 0, 0}, CUBE_GLOBAL_MIN = {1, 1, 1};
    cube cubes[MAX_CUBE];
    
    srand((unsigned int) time(0));

    cube_number=read_input(inpath,cubes,&CUBE_GLOBAL_MAX,&CUBE_GLOBAL_MIN);

    printf("-- Transient Mode analysis ... \n");

    // define the rays
    //ray ray_traj = rand_ray(CUBE_GLOBAL_MIN, CUBE_GLOBAL_MAX, Linear);
    point3d ray_start = {699, 1256, CUBE_GLOBAL_MAX.z};
    point3d ray_end = {699, 1256, CUBE_GLOBAL_MIN.z};
    ray ray_traj = fixed_ray(ray_start, ray_end, Constant);
    
    FILE *fout = fopen("out.txt", "w");

    //float ray_dist = distance(ray_traj.start, ray_traj.end);
    point3d curr_ray_pos;
    int point_ray_dist;
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
            for(int point_index = 0; point_index < cubes[cube_index].point_amt; point_index++){  //Iterates over points in the cube    
                curr_ray_pos = ray_traj.start;                  
                for(int step = 0; step < N_STEPS; step++){                                //Iterates over ray steps
                    point_ray_dist = distance(cubes[cube_index].points[point_index].pos, curr_ray_pos);
                    cubes[cube_index].points[point_index].energy[step + 1] = 
                        cubes[cube_index].points[point_index].energy[step] +
                        bell(0, 130, point_ray_dist) *
                        1000 *
                        ray_traj.energy_curve[step];
                    
                    curr_ray_pos.x += ray_traj.delta.x;
                    curr_ray_pos.y += ray_traj.delta.y;
                    curr_ray_pos.z += ray_traj.delta.z;
                }
                fprintf(fout, "%f,%f,%f", cubes[cube_index].points[point_index].pos.x, cubes[cube_index].points[point_index].pos.y, cubes[cube_index].points[point_index].pos.z);
                for (int step=1; step <= N_STEPS; step++) fprintf(fout, ",%f", cubes[cube_index].points[point_index].energy[step]);
                fprintf(fout, "\n");
                cube_energy += cubes[cube_index].points[point_index].energy[N_STEPS];
            }
            printf("Energia: %f\n", cube_energy);
        }
    }
    free_cubes(cubes, cube_number);
    //fclose(fin);
    fclose(fout);
    return 0;
    
}
//AGGIUNGERE free()!!