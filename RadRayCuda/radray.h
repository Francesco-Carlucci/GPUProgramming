#include "3dmisc.h"

#ifndef RADRAY_H
#define RADRAY_H

#define MAX_CUBE 300
#define ENERGY_CURVE_SIZE 100
#define N_STEPS 100
#define MAX_POINTS 1000
#define N_RAYS 3

typedef enum energy_type_e{
    Constant,
    Linear,
    Bragg,
}energy_type;

typedef struct energy_point_s{
    point3d pos;
    double energy[N_STEPS + 1];  //[N_STEPS]
}energy_point;

typedef struct cube_s{
    point2d* limits;
    int N;
    point3d min;
    point3d max;
    int layer_n;
    energy_point* points;
    int point_amt;
}cube;

typedef struct ray_s{
    point3d start;
    point3d end;
    point3d delta;
    int steps;
    energy_type profile;
    float energy_curve[ENERGY_CURVE_SIZE];
}ray;

int cube_contains_point(cube cu, point3d p);
int point_in_polygon(cube poly,point3d p);
int cube_contains_ray(cube cu, ray r);
ray rand_ray(point3d bound_min, point3d bound_max, energy_type profile);
ray fixed_ray(point3d start, point3d end, energy_type profile);
void generate_energy_profile(ray *ray);
void generate_points_by_amount(cube *curr_cube, int amount);
void generate_points_by_resolution(cube *curr_cube, point3d resolution);
void generate_points_by_resolution_parallel(cube *curr_cube, point3d resolution,energy_point** dev_points);
void free_cube(cube *cu);
void free_cubes(cube *c_arr, int n);
void generate_rays(ray ray_arr[], ray main_ray, int amount);
int read_input(char* inpath,cube cubes[],point3d* CUBE_GLOBAL_MAX, point3d* CUBE_GLOBAL_MIN);
void write_on_file(FILE *fout, cube *cubes, int cube_number, ray ray_traj);

#endif