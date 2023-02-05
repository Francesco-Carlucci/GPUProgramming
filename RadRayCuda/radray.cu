#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "radray.h"

#define MAX_THREADS 1024


/* Function used to check if the point is inside the cube */
__device__ int point_in_polygon_dev(point2d *limits, int N, float minz, float maxz ,point3d p){
    int inside=0;
    point2d p1,p2;
    p1 = limits[0];

    for(int i=1;i<N;i++){
        p2=limits[i%N];
        printf("%f %f %f\n", p.z, minz, maxz);
        if((p.y>min(p1.y,p2.y)) &&(p.y<=max(p1.y,p2.y))&&(p.x<=max(p1.x,p2.x))&&(p1.y != p2.y)){
            if (p.x<(p.y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x && p.z>=minz && p.z<=maxz){
                inside=!inside;
            }
        }
        p1=p2;
    }
    return inside;
}


/* Kernel lauched to initialize the points of the cube hit by the ray. Each threads has a specific point in the area of the cube, if the point is contained in the cube itself (not only 
in the area surrounding the cube) the point is initialized.
TODO: maybe create struct for the parameters or something like that to reduce the lenght of function call
*/
__global__ void initialize_points(point2d *limits, energy_point *points, int x_amt, int y_amt, int z_amt, float minx, float miny, float minz, int dx, int dy, int dz, int N, float maxz) {

    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    int x = tid%x_amt;
    int y = (tid/x_amt)%y_amt;
    int z = tid/(x_amt*y_amt);
    point3d t;

    // with this condition we exclude the thread in excess
    if (z<z_amt) {
        t.x = minx + x * dx;
        t.y = miny + y * dy;
        t.z = minz + z * dz;
        //printf("%f %f %f\n", t.x, t.y, t.z);
        if (point_in_polygon_dev(limits, N, minz, maxz, t)) {
            //printf("%d %d %d %f %f\n", z, z_amt, dz, t.z, minz);
            //printf("%f %f %f\n", t.x, t.y, t.z);
            points[tid].pos = t;
            points[tid].energy[0] = 0;
            points[tid].energy[N_STEPS] = 0;
        }
    }

}


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

int cube_contains_point(cube cu, point3d p){
    if(p.x>=cu.min.x && p.y>=cu.min.y && p.z>=cu.min.z
    && p.x<=cu.max.x && p.y<=cu.max.y && p.z<=cu.max.z){
        return 1;
    }
    return 0;
}

int point_in_polygon(cube poly,point3d p){
    int inside=0;
    point2d p1,p2;
    p1=poly.limits[0];
    for(int i=1;i<poly.N;i++){
        p2=poly.limits[i%poly.N];
        printf("%f %f %f\n", p.z, poly.min.z, poly.max.z);
        if((p.y>min(p1.y,p2.y)) &&(p.y<=max(p1.y,p2.y))&&(p.x<=max(p1.x,p2.x))&&(p1.y != p2.y)){
            if (p.x<(p.y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x && p.z>=poly.min.z && p.z<=poly.max.z){
                inside=!inside;
            }
        }
        p1=p2;
    }
    return inside;
}

int cube_contains_ray(cube cu, ray r) {
    point3d ray_pos = r.start;
    for (ray_pos.z = cu.max.z; ray_pos.z > cu.min.z; ray_pos.z+=r.delta.z) {    //rendere for da cu.min.z a cu.max.z
        if (point_in_polygon(cu, ray_pos)) return 1;
        ray_pos.x += r.delta.x;
        ray_pos.y += r.delta.y;
        //ray_pos.z += r.delta.z;
    }
    return 0;
}

ray rand_ray(point3d bound_min, point3d bound_max, energy_type profile) {     //randomly generates the trajectory of a ray given the box bounds
    ray r;
    r.profile = profile;
    r.start.x = rand_unit() * (bound_max.x - bound_min.x) + bound_min.x;
    r.start.y = rand_unit() * (bound_max.y - bound_min.y) + bound_min.y;
    r.start.z = bound_max.z;
    r.end.x = rand_unit() * (bound_max.x - bound_min.x) + bound_min.x;
    r.end.y = rand_unit() * (bound_max.y - bound_min.y) + bound_min.y;
    r.end.z = bound_min.z;
    r.delta.x = (r.end.x - r.start.x) / N_STEPS;
    r.delta.y = (r.end.y - r.start.y) / N_STEPS;
    r.delta.z = (r.end.z - r.start.z) / N_STEPS;
    r.steps = N_STEPS;
    generate_energy_profile(&r);
    return r;
}

ray fixed_ray(point3d start, point3d end, energy_type profile) {     //generates the trajectory of a ray given the start and end positions
    ray r;
    r.profile = profile;
    r.start = start;
    r.end = end;
    r.delta.x = (r.end.x - r.start.x) / N_STEPS;
    r.delta.y = (r.end.y - r.start.y) / N_STEPS;
    r.delta.z = (r.end.z - r.start.z) / N_STEPS;
    r.steps = N_STEPS;
    generate_energy_profile(&r);
    return r;
}

void generate_energy_profile(ray *ray) {    //Aggiungere scalamento in base alla lunghezza del raggio
    switch (ray->profile) {
        case Bragg:
            //TODO
            for (int i = 0; i < ENERGY_CURVE_SIZE; i++) {
                ;
            }
            break;
        
        case Linear:
            for (int i = 0; i < ENERGY_CURVE_SIZE; i++) {
                ray->energy_curve[i] = 1 - i/ENERGY_CURVE_SIZE * i;
            }
            break;
        
        case Constant:
        default:
            for (int i = 0; i < ENERGY_CURVE_SIZE; i++) {
                ray->energy_curve[i] = 1;
            }
            break;
    }
    return;
}

void generate_points_by_amount(cube *curr_cube, int amount){  //generates amount points on a grid in each box
    /***
     * Can be improved by generating only points in the boundary or only points near the ray.
     * */
    int cubroot = cbrtf(MAX_POINTS);
    point3d t;
    int dx = (curr_cube->max.x - curr_cube->min.x) / cubroot;
    int dy = (curr_cube->max.y - curr_cube->min.y) / cubroot;
    int dz = (curr_cube->max.z - curr_cube->min.z) / cubroot;
    int cnt=0;
    curr_cube->points = (energy_point *) malloc(MAX_POINTS * sizeof(energy_point));
    for(int i = 0; i < cubroot; i++){
        for(int j = 0; j < cubroot; j++){
            for(int k = 0; k < cubroot; k++){
                t.x = curr_cube->min.x + i * dx;
                t.y = curr_cube->min.y + j * dy;
                t.z = curr_cube->min.z + k * dz;
                curr_cube->points[cnt].pos = t;
                curr_cube->points[cnt].energy[0] = 0;
                curr_cube->points[cnt].energy[N_STEPS] = 0;
                cnt++;
            }
        }
    }
    curr_cube->point_amt = cnt;
    return;
}

void generate_points_by_resolution(cube *curr_cube, point3d resolution){  //generates MAX_POINTS points on a grid in each box
    point3d t;
    int cnt = 0;
    int dx = resolution.x;
    int dy = resolution.y;
    int dz = resolution.z;
    int x_amt = (curr_cube->max.x - curr_cube->min.x) / dx;
    int y_amt = (curr_cube->max.y - curr_cube->min.y) / dy;
    int z_amt = (curr_cube->max.z - curr_cube->min.z) / dz;
    curr_cube->points = (energy_point *) malloc(x_amt * y_amt * z_amt * sizeof(energy_point));  //nvcc vuole il cast
    for(int i = 0; i < x_amt; i++){
        for(int j = 0; j < y_amt; j++){
            for(int k = 0; k < z_amt; k++){
                t.x = curr_cube->min.x + i * dx;
                t.y = curr_cube->min.y + j * dy;
                t.z = curr_cube->min.z + k * dz;
                //printf("%f %f %f\n", t.x, t.y, t.z);
                if (point_in_polygon(*curr_cube, t)) {
                    //printf("%d %d %d %f %f\n", k, z_amt, dz, t.z, curr_cube->min.z);
                    //printf("%f %f %f\n", t.x, t.y, t.z);
                    curr_cube->points[cnt].pos = t;
                    curr_cube->points[cnt].energy[0] = 0;
                    curr_cube->points[cnt].energy[N_STEPS] = 0;
                    cnt++;
                }
            }
        }
    }
    curr_cube->point_amt = cnt;
    return;
}


/* Generates the points inside a cube with a given resolution, it directly returns the pointer to the points vector in the GPU so that we don't need to
   copy it back to host and again to the GPU */
energy_point* generate_points_by_resolution_parallel(cube *curr_cube, point3d resolution){

    // data structures used
    point3d t;
    int cnt = 0;
    int dx = resolution.x;
    int dy = resolution.y;
    int dz = resolution.z;
    int x_amt = (curr_cube->max.x - curr_cube->min.x) / dx;
    int y_amt = (curr_cube->max.y - curr_cube->min.y) / dy;
    int z_amt = (curr_cube->max.z - curr_cube->min.z) / dz;
    curr_cube->point_amt = x_amt*y_amt*z_amt;
    point2d *dev_limits;
    energy_point *dev_points;
    curr_cube->points = (energy_point *) malloc(curr_cube->point_amt * sizeof(energy_point));  //nvcc vuole il cast

    // blocks needed to cover all possible points
    int nblocks = (curr_cube->point_amt)/MAX_THREADS+1;
    // allocation of data structures for GPU
    cudaMalloc( (void**) &dev_limits, curr_cube->N * sizeof(point2d));
    cudaMalloc( (void**) &dev_points, curr_cube->point_amt * sizeof(energy_point));
    // copy of limits array
    cudaMemcpy(dev_limits, curr_cube->limits, curr_cube->N * sizeof(point2d), cudaMemcpyHostToDevice);
    initialize_points<<<nblocks,MAX_THREADS>>>(dev_limits, dev_points, x_amt, y_amt, z_amt, curr_cube->min.x, curr_cube->min.y, curr_cube->min.z, dx, dy, dz, curr_cube->N, curr_cube->min.z);
    //cudaMemcpy(curr_cube->points, dev_points, curr_cube->point_amt * sizeof(energy_point), cudaMemcpyDeviceToHost);
    // free structures
    cudaFree(dev_limits); 

    return dev_points;
}



void free_cube(cube *cu) {
    if(cu->points!=NULL){
        free(cu->points);
    }
    free(cu->limits);
    return;
}

void free_cubes(cube *c_arr, int n) {
    for (int i = 0; i < n; i++) {
        free_cube(&c_arr[i]);
    }
    return;
}

void generate_rays(ray ray_arr[], ray main_ray, int amount) {
    point3d new_start, new_end, new_delta, ang_coeff;
    int new_steps;
    float main_ray_factor;
    float new_length;
    float norm;
    
    ray_arr[0] = main_ray;
    for (int i = 1; i < amount; i++) {
        main_ray_factor = rand_unit();
        new_start.x = main_ray.start.x + main_ray_factor * (main_ray.end.x - main_ray.start.x);
        new_start.y = main_ray.start.y + main_ray_factor * (main_ray.end.y - main_ray.start.y);
        new_start.z = main_ray.start.z + main_ray_factor * (main_ray.end.z - main_ray.start.z);
        ang_coeff.x = rand_unit();
        ang_coeff.y = rand_unit();
        ang_coeff.z = rand_unit();
        norm = sqrt(pow(ang_coeff.x,2) + pow(ang_coeff.y,2) + pow(ang_coeff.z,2));
        new_delta.x = ang_coeff.x / norm;
        new_delta.y = ang_coeff.y / norm;
        new_delta.z = ang_coeff.z / norm;
        new_steps = rand_unit() * main_ray.steps;
        new_end.x = new_start.x + new_delta.x * new_steps;
        new_end.y = new_start.y + new_delta.y * new_steps;
        new_end.z = new_start.z + new_delta.z * new_steps;

        ray_arr[i].start = new_start;
        ray_arr[i].end = new_end;
        ray_arr[i].delta = new_delta;
        ray_arr[i].steps = new_steps;
    }
    return;
}


