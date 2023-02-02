#include <stdlib.h>
#include <math.h>

#include "3dmisc.h"

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
    float dist;
    dist = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
    return dist;
}

double bell(float mu, float sigma, float dist) {
    //double bell_value=((1 / (2.506628274631 * sigma)) * exp(-0.5 * pow((1 / sigma * (dist - mu)), 2)));
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