#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_CUBE 300
#define ENERGY_CURVE_SIZE 100
#define N_STEPS 100
#define MAX_POINTS 1000

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

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
    float x;
    float y;
}point2d;

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
    float energy_curve[ENERGY_CURVE_SIZE];
    //float energy;
}ray;

point3d rand_point(point3d min, point3d max) {
    point3d t;
    t.x = rand() / RAND_MAX * (max.x - min.x) + min.x;
    t.y = rand() / RAND_MAX * (max.y - min.y) + min.y;
    t.z = rand() / RAND_MAX * (max.z - min.z) + min.z;
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
    trajectory.start.x = rand() / (float)RAND_MAX * (bound_max.x - bound_min.x) + bound_min.x;
    trajectory.start.y = rand() / (float)RAND_MAX * (bound_max.y - bound_min.y) + bound_min.y;
    trajectory.start.z = bound_max.z;
    trajectory.end.x = rand() / (float)RAND_MAX * (bound_max.x - bound_min.x) + bound_min.x;
    trajectory.end.y = rand() / (float)RAND_MAX * (bound_max.y - bound_min.y) + bound_min.y;
    trajectory.end.z = bound_min.z;
    trajectory.delta.x = (trajectory.end.x - trajectory.start.x) / N_STEPS;
    trajectory.delta.y = (trajectory.end.y - trajectory.start.y) / N_STEPS;
    trajectory.delta.z = (trajectory.end.z - trajectory.start.z) / N_STEPS;
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

int cube_contains_point(cube cu, point3d p){
    if(p.x>=cu.min.x && p.y>=cu.min.y && p.z>=cu.min.z
    && p.x<=cu.max.x && p.y<=cu.max.y && p.z<=cu.max.z){
        return 1;
    }
    return 0;
}

int point_in_poligon(cube poly,point3d p){
    int inside=0;
    point2d p1,p2;
    p1=poly.limits[0];
    for(int i=1;i<poly.N;i++){
        p2=poly.limits[i%poly.N];
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
        if (point_in_poligon(cu, ray_pos)) return 1;
        ray_pos.x += r.delta.x;
        ray_pos.y += r.delta.y;
        //ray_pos.z += r.delta.z;
    }
    return 0;
}

int main(int argc, char* argv[]){
    FILE* fin;
    char* inpath = "../RadrayPy/out_all_points.txt";
    int i; //i é l'indice del cubo
    point3d CUBE_GLOBAL_MAX = {0, 0, 0}, CUBE_GLOBAL_MIN = {1, 1, 1};
    cube cubes[MAX_CUBE];
    //polygon polygons[MAX_CUBE];
    //cube t;
    cube t;
    
    srand((unsigned int) time(0));

    fin = fopen(inpath, "r");
    if(fin==NULL){
        printf("Unable to read the file!");
        exit(1);
    }

    int cube_number=0;
    /***
     * Read file, format for each boundary:
     * N    layer
     * minz maxz
     * x    y   (repeated N times)
     * Calculates minx and miny of the cube Bounding Box
     * Calculates Global bounding box
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
        if (t.max.x > CUBE_GLOBAL_MAX.x) { CUBE_GLOBAL_MAX.x = t.max.x; }  // computes global max and min, in which the ray must pass
        if (t.max.y > CUBE_GLOBAL_MAX.y) { CUBE_GLOBAL_MAX.y = t.max.y; }
        if (t.max.z > CUBE_GLOBAL_MAX.z) { CUBE_GLOBAL_MAX.z = t.max.z; }
        if (t.min.x < CUBE_GLOBAL_MIN.x) { CUBE_GLOBAL_MIN.x = t.min.x; }
        if (t.min.y < CUBE_GLOBAL_MIN.y) { CUBE_GLOBAL_MIN.y = t.min.y; }
        if (t.min.z < CUBE_GLOBAL_MIN.z) { CUBE_GLOBAL_MIN.z = t.min.z; }
        cubes[cube_number] = t;
        cube_number++;
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
    float dist_threshold = 10000;       //=max_x_ray

    float cube_energy;

    //printf("%f,%f,%f,%f,%f,%f\n", ray_traj.start.x, ray_traj.start.y, ray_traj.start.z, ray_traj.end.x, ray_traj.end.y, ray_traj.end.z);
    fprintf(fout, "%f,%f,%f,%f,%f,%f\n", ray_traj.start.x, ray_traj.start.y, ray_traj.start.z, ray_traj.end.x, ray_traj.end.y, ray_traj.end.z);
    for(int cube_index = 0; cube_index < cube_number; cube_index++){               //Iterates over the cubes
        if(cube_contains_ray(cubes[cube_index], ray_traj)){                     //Check if the ray is in the cube
            cube_energy = 0;
            printf("Raggio in cubo %d - ", cube_index);
            fprintf(fout, "%d\n", cube_index);
            generate_points_by_amount(&cubes[cube_index], MAX_POINTS);
            //generate_points_by_resolution(&cubes[cube_index], res);
            for(int point_index = 0; point_index < cubes[cube_index].point_amt; point_index++){  //Iterates over points in the cube    
                curr_ray_pos = ray_traj.start;                  
                for(int step = 0; step < N_STEPS; step++){                                //Iterates over ray steps
                    point_ray_dist = distance(cubes[cube_index].points[point_index].pos, curr_ray_pos);
                    //if (point_ray_dist < dist_threshold) {
                        cubes[cube_index].points[point_index].energy[step + 1] = cubes[cube_index].points[point_index].energy[step] + bell(0, 130, point_ray_dist) * 1000; //* 1000 * CONSTANT[N_STEPS]/100;
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