#include "3dmisc.h"

#ifndef RADRAY_H
#define RADRAY_H

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
    point3d pos;
    double energy[N_STEPS + 1];  //[N_STEPS]
}energy_point;

typedef struct {
    point2d* limits;
    int N;
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

int cube_contains_point(cube cu, point3d p);
int point_in_polygon(cube poly,point3d p);
int cube_contains_ray(cube cu, ray r);
ray rand_ray(point3d bound_min, point3d bound_max);
ray fixed_ray(point3d start, point3d end);
void generate_points_by_amount(cube *curr_cube, int amount);
void generate_points_by_resolution(cube *curr_cube, point3d resolution);
void free_cube(cube *cu);
void free_cubes(cube *c_arr, int n);
void generate_rays(ray ray_arr[], ray main_ray, int amount);

#endif