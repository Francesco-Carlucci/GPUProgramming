#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define MAX_CUBE 300
#define ENERGY_CURVE_SIZE 100
#define N_STEPS 100
#define MAX_POINTS 1000

static char LINEAR[ENERGY_CURVE_SIZE] = {
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100
};

typedef struct {
    float x;
    float y;
    float z;
}point3d;

typedef struct {
    point3d min;
    point3d max;
    int layer_n;
    point3d points[MAX_POINTS];
} cube;

typedef struct {
    point3d start;
    point3d end;
    point3d delta;
    float energy_curve[ENERGY_CURVE_SIZE];
    //float energy;
} ray;

point3d rand_point(point3d min, point3d max) {
    point3d t;
    t.x = rand() * (max.x - min.x) + min.x;
    t.y = rand() * (max.y - min.y) + min.y;
    t.z = rand() * (max.z - min.z) + min.z;
    return t;
}

float distance(point3d a, point3d b) {  //compute distance between two points
    int dist;
    dist = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
    return dist;
}

point3d param_to_coord(point3d start, point3d end, float t) {  //return x,y of a line given its parameter t
    point3d p;
    p.x = (end.x - start.x) * t;
    p.y = (end.y - start.y) * t;
    p.z = (end.z - start.z) * t;
    return p;
}

ray rand_ray(point3d min, point3d max) {     //randomly generates the trajectory of a ray given the box bounds
    ray trajectory;
    memcpy(LINEAR, trajectory.energy_curve,ENERGY_CURVE_SIZE*sizeof(char));
    trajectory.start.x = rand() * (max.x - min.x) + min.x;
    trajectory.start.y = rand() * (max.y - min.y) + min.y;
    trajectory.start.z = max.z;
    trajectory.end.x = rand() * (max.x - min.x) + min.x;
    trajectory.end.y = rand() * (max.y - min.y) + min.y;
    trajectory.end.z = min.z;
    trajectory.delta.x = (trajectory.end.x - trajectory.start.x) / N_STEPS;
    trajectory.delta.y = (trajectory.end.y - trajectory.start.y) / N_STEPS;
    trajectory.delta.z = (trajectory.end.z - trajectory.start.z) / N_STEPS;
    return trajectory;
}

void generate_points(cube curr_cube){  //generates MAX_POINTS points on a grid in each box
    /***
     * TODO: can be improved by generating only points in the boundary or only points near the ray.
     * */
    int cubroot = cbrtf(MAX_POINTS);
    point3d t;
    int dx = (curr_cube.max.x - curr_cube.min.x) / cubroot;
    int dy = (curr_cube.max.y - curr_cube.min.y) / cubroot;
    int dz = (curr_cube.max.z - curr_cube.min.z) / cubroot;
    int cnt=0;
    for(int i = 0; i < cubroot; i++){
        for(int j = 0; j < cubroot; j++){
            for(int k = 0; k < cubroot; k++){
                t.x = curr_cube.min.x + i * dx;
                t.y = curr_cube.min.y + j * dy;
                t.z = curr_cube.min.z + k * dz;
                curr_cube.points[cnt] = t;
                cnt++;
            }
        }
    }
}
int contains(cube curr_cube,point3d p){
    if((p.x>=curr_cube.min.x) && (p.y>=curr_cube.min.y) && p.z>=curr_cube.min.z
    && p.x<=curr_cube.max.x && p.y<=curr_cube.max.y && p.z<=curr_cube.max.z){
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[]){
    FILE* fin;
    char* inpath = "../RadrayPy/output.txt";
    int i; //i é l'indice del cubo
    point3d CUBE_GLOBAL_MAX = {0, 0, 0}, CUBE_GLOBAL_MIN = {FLT_MAX, FLT_MAX, FLT_MAX};
    cube cubes[MAX_CUBE];
    cube t;
    
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
    printf("-- Transient Mode analysis ... ");
    
    ray ray_traj = rand_ray(CUBE_GLOBAL_MAX, CUBE_GLOBAL_MIN);
    
    //float ray_dist = distance(ray_traj.start, ray_traj.end);
    point3d curr_ray_p;

    for(int cube_index = 0; cube_index < MAX_CUBE; cube_index++){            //Iterates over the cubes
        for(int point_index = 0; point_index < MAX_POINTS; point_index++){   //Iterates over points in the cube
            curr_ray_p = ray_traj.start;
            for(int t = 0; t < N_STEPS; t++){                                //Iterates over ray steps
                curr_ray_p.x+=ray_traj.delta.x;
                curr_ray_p.y+=ray_traj.delta.y;
                curr_ray_p.z+=ray_traj.delta.z;
                //check if the ray is in the cube
                if(contains(cubes[cube_index],curr_ray_p)){
                    int p_ray_dist= distance(cubes[cube_index].points[point_index],curr_ray_p);
                    
                }
            }
        }
    }

    return 0;
    
}
