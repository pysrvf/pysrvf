//#include "params.h"
#include "shape.h"
#include <stdio.h>
#include <vector>

//float InnerProduct(float *, float *, float *, float *, params *, float * );

//void GeodesicDistance(shape &, shape &, shape &, shape &, params *, GeodesicData *);

//bool Read_Theta_fn(char *,shape **,shape **,float *, float *);

bool Read_v1_v2(char *,shape **,shape **,float *, float *);
bool Read_q1_q2(char *,shape **,shape **);
bool Read_X1_X2(char *,shape **,shape **);


float MatchQ(shape *, shape *, float *, float **);
float MatchCoords(shape *, shape *, float *, float **);

//float MatchPaths(shape *, shape *,float , float , float *, float **);

//float Match_Cost(shape *, shape *,int ,int ,int ,int ,float ,float );

//float Match_CostV(shape *, shape *,int ,int ,int ,int ,float ,float );

float Match_CostQ(shape *, shape *,int ,int ,int ,int);
float Match_Cost_Coords(shape *, shape *,int ,int ,int ,int);


//float Match_CostV(float *, float *,shape *v_ps1, shape *v_ps2,int k,int l,int i,int j,float a ,float b);

//float Cost_Group_Action_By_Gamma(float *, float *, float *, float *, float *,int, float , float );

//float L2_Costfn(float *, float *, float *, float *,int, float ,float );


float norm(float *v_pfArray,int v_ilen);

bool WriteGamma(FILE * ,float *,int, shape *);

bool WriteGammaV(FILE * ,float *,int, shape *);

void linint(float *xnew, float *ynew, int cnt, float *xx, float *yy, int n);

void splint(float *xa, float *ya, float *y2a, int n, float x, float *y);

void spline(float *x, float *y, int n, float yp1, float ypn, float *y2);
