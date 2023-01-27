#ifndef _3D_MISC_H
#define _3D_MISC_H

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

typedef struct point3d_{
    float x;
    float y;
    float z;
}point3d;

typedef struct point2d_{
    float x;
    float y;
}point2d;

float rand_unit();
point3d rand_point(point3d min, point3d max);
float distance(point3d a, point3d b);
double bell(float mu, float sigma, float dist);
double norm_bell(float mu, float sigma, float dist);
point3d param_to_coord(point3d start, point3d end, float t);

#endif