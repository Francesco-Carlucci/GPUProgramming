#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <math.h>

#define MAX_CUBE 300
#define ENERGY_CURVE_SIZE 100
#define N_STEPS 100
#define MAX_POINTS 1000

#define N_RAYS 3

static char CONSTANT[ENERGY_CURVE_SIZE] = {
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
};

typedef struct {
    float x;
    float y;
    float z;
}point3d;

typedef struct {
    point3d pos;
    double energy[N_STEPS + 1];  //[N_STEPS]
}energy_point;

typedef struct {
    point3d min;
    point3d max;
    int layer_n;
    energy_point* points;
    int point_amt;
}cube;

typedef struct {
    point3d start;
    point3d end;
    point3d delta;
    int steps;
    float energy_curve[ENERGY_CURVE_SIZE];
    //float energy;
}ray;

float rand_unit() {
    return rand() / (float)RAND_MAX;
}

point3d rand_point(point3d min, point3d max) {
    point3d t;
    t.x = rand_unit() * (max.x - min.x) + min.x;
    t.y = rand_unit() * (max.y - min.y) + min.y;
    t.z = rand_unit() * (max.z - min.z) + min.z;
    return t;
}



float distance(point3d a, point3d b) {  //compute distance between two points
    int dist;
    dist = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
    return dist;
}

double bell(float mu, float sigma, float dist) {
    return ((1 / (2.506628274631 * sigma)) * exp(-0.5 * pow((1 / sigma * (dist - mu)), 2)));
    //return ((1 / (sqrt(2 * M_PI) * sigma)) *
    //exp(-0.5 * pow((1 / sigma * (dist - mu)), 2)));
}

double norm_bell(float mu, float sigma, float dist) {
    double p = pow((dist / 10 - mu), 2);
    return exp(-0.5 * p);
}

point3d param_to_coord(point3d start, point3d end, float t) {  //return x,y of a line given its parameter t
    point3d p;
    p.x = (end.x - start.x) * t;
    p.y = (end.y - start.y) * t;
    p.z = (end.z - start.z) * t;
    return p;
}

ray rand_ray(point3d bound_min, point3d bound_max) {     //randomly generates the trajectory of a ray given the box bounds
    ray trajectory;
    //memcpy(CONSTANT, trajectory.energy_curve,ENERGY_CURVE_SIZE*sizeof(char));
    trajectory.start.x = rand_unit() * (bound_max.x - bound_min.x) + bound_min.x;
    trajectory.start.y = rand_unit() * (bound_max.y - bound_min.y) + bound_min.y;
    trajectory.start.z = bound_max.z;
    trajectory.end.x = rand_unit() * (bound_max.x - bound_min.x) + bound_min.x;
    trajectory.end.y = rand_unit() * (bound_max.y - bound_min.y) + bound_min.y;
    trajectory.end.z = bound_min.z;
    trajectory.delta.x = (trajectory.end.x - trajectory.start.x) / N_STEPS;
    trajectory.delta.y = (trajectory.end.y - trajectory.start.y) / N_STEPS;
    trajectory.delta.z = (trajectory.end.z - trajectory.start.z) / N_STEPS;
    trajectory.steps = N_STEPS;
    return trajectory;
}

ray fixed_ray(point3d start, point3d end) {     //generates the trajectory of a ray given the start and end positions
    ray trajectory;
    //memcpy(CONSTANT, trajectory.energy_curve,ENERGY_CURVE_SIZE*sizeof(char));
    trajectory.start = start;
    trajectory.end = end;
    trajectory.delta.x = (trajectory.end.x - trajectory.start.x) / N_STEPS;
    trajectory.delta.y = (trajectory.end.y - trajectory.start.y) / N_STEPS;
    trajectory.delta.z = (trajectory.end.z - trajectory.start.z) / N_STEPS;
    trajectory.steps = N_STEPS;
    return trajectory;
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
    curr_cube->points = malloc(MAX_POINTS * sizeof(energy_point));
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
    curr_cube->points = malloc(x_amt * y_amt * z_amt * sizeof(energy_point));
    for(int i = 0; i < x_amt; i++){
        for(int j = 0; j < y_amt; j++){
            for(int k = 0; k < z_amt; k++){
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
}

int cube_contains_point(cube cu, point3d p){
    if(p.x>=cu.min.x && p.y>=cu.min.y && p.z>=cu.min.z
    && p.x<=cu.max.x && p.y<=cu.max.y && p.z<=cu.max.z){
        return 1;
    }
    return 0;
}

int cube_contains_ray(cube cu, ray r) {
    point3d ray_pos = r.start;
    for (int i = 0; i < N_STEPS; i++) {
        if (cube_contains_point(cu, ray_pos)) return 1;
        ray_pos.x += r.delta.x;
        ray_pos.y += r.delta.y;
        ray_pos.z += r.delta.z;
    }
    return 0;
}

int main(int argc, char* argv[]){
    FILE* fin;
    char* inpath = "../RadrayPy/output.txt";
    int i; //i é l'indice del cubo
    point3d CUBE_GLOBAL_MAX = {0, 0, 0}, CUBE_GLOBAL_MIN = {1, 1, 1};
    cube cubes[MAX_CUBE];
    cube t;
    
    srand(123456);

    fin = fopen(inpath, "r");  //file input, contains min and max of each cube in the GDS (2 x,y,z each)
    
    for (i = 0; fscanf(fin, "%f,%f,%f,%f,%f,%f,%d", &t.min.x, &t.min.y, &t.min.z, &t.max.x, &t.max.y, &t.max.z, &t.layer_n) != EOF; i++) {
        cubes[i] = t;
        if (t.max.x > CUBE_GLOBAL_MAX.x) {CUBE_GLOBAL_MAX.x = t.max.x;} /* computes global max and min, in which the ray must pass */
        if (t.max.y > CUBE_GLOBAL_MAX.y) {CUBE_GLOBAL_MAX.y = t.max.y;}
        if (t.max.z > CUBE_GLOBAL_MAX.z) {CUBE_GLOBAL_MAX.z = t.max.z;}
        if (t.min.x < CUBE_GLOBAL_MIN.x) {CUBE_GLOBAL_MIN.x = t.min.x;}
        if (t.min.y < CUBE_GLOBAL_MIN.y) {CUBE_GLOBAL_MIN.y = t.min.y;}
        if (t.min.z < CUBE_GLOBAL_MIN.z) {CUBE_GLOBAL_MIN.z = t.min.z;}
    }
    printf("-- Transient Mode analysis ... \n");

    //ray ray_traj = rand_ray(CUBE_GLOBAL_MIN, CUBE_GLOBAL_MAX);
    point3d ray_start = {699, 1256, CUBE_GLOBAL_MAX.z};
    point3d ray_end = {699, 1256, CUBE_GLOBAL_MIN.z};
    ray ray_traj = fixed_ray(ray_start, ray_end);
    
    FILE *fout = fopen("out.txt", "w");

    //float ray_dist = distance(ray_traj.start, ray_traj.end);
    point3d curr_ray_pos;
    int point_ray_dist;
    point3d res = {10, 10, 10};
    float dist_threshold = 10000;

    float cube_energy;

    ray ray_arr[N_RAYS];
    generate_rays(ray_arr, ray_traj, N_RAYS);

    fprintf(fout, "%d\n", N_RAYS);
    for (int i = 0; i < N_RAYS; i++) {
        fprintf(fout, "%f,%f,%f,%f,%f,%f\n", ray_arr[i].start.x, ray_arr[i].start.y, ray_arr[i].start.z, ray_arr[i].end.x, ray_arr[i].end.y, ray_arr[i].end.z);
    }
    for(int cube_index = 0; cube_index < i; cube_index++){               //Iterates over the cubes
        if(cube_contains_ray(cubes[cube_index], ray_traj)){                     //Check if the ray is in the cube
            cube_energy = 0;
            printf("Raggio in cubo %d - ", cube_index);
            fprintf(fout, "%d\n", cube_index);
            generate_points_by_amount(&cubes[cube_index], MAX_POINTS);
            //generate_points_by_resolution(&cubes[cube_index], res);
            for(int point_index = 0; point_index < cubes[cube_index].point_amt; point_index++){  //Iterates over points in the cube    
                curr_ray_pos = ray_traj.start;                  
                for(int t = 0; t < N_STEPS; t++){                                //Iterates over ray steps
                    point_ray_dist = distance(cubes[cube_index].points[point_index].pos, curr_ray_pos);
                    //if (point_ray_dist < dist_threshold) {
                        cubes[cube_index].points[point_index].energy[t+1] = cubes[cube_index].points[point_index].energy[t] + bell(0, 130, point_ray_dist) * 1000; //* 1000 * CONSTANT[N_STEPS]/100;
                        curr_ray_pos.x += ray_traj.delta.x;
                        curr_ray_pos.y += ray_traj.delta.y;
                        curr_ray_pos.z += ray_traj.delta.z;
                    //}
                }
                fprintf(fout, "%f,%f,%f,%f\n", cubes[cube_index].points[point_index].pos.x, cubes[cube_index].points[point_index].pos.y, cubes[cube_index].points[point_index].pos.z, cubes[cube_index].points[point_index].energy[N_STEPS]);
                cube_energy += cubes[cube_index].points[point_index].energy[N_STEPS];
            }
            printf("Energia: %f\n", cube_energy);
        }
    }
    fclose(fin);
    fclose(fout);
    return 0;
    
}
//AGGIUNGERE free()!!