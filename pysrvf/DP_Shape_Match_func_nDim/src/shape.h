#if !defined(SHAPE_H)
#define SHAPE_H
#include <math.h>
//#include "params.h"

const float PI =  (float)3.14159265358979;

class shape
{
	public:
		//Member variables
		float *m_pfPhi;
		float *m_pfTheta;
		float *m_v11;
		float *m_v12;
        float *m_v13;
                
		float **v;
		float *h;
		int m_iT;
		int m_n;
		//Member functions
		shape();//default constructor
		shape(int, int);
		shape(int);
		shape(int, float *,float *);	
		shape(shape &);		
		~shape();
		shape& operator = (const shape &s);
		void create(int);
		void create(int, float *, float *);
		void Initalize(int ,float *, float *);
		float L2_Costfn(shape &);
		void copy(int T, float *, float *);
//		float arr[29696][239];
		float **arr;
	
		
		void Create_From_Coordinates(float *, float *,  int);
};
#endif
 