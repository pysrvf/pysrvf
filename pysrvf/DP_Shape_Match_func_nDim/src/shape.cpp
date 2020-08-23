#include <stdlib.h>
#include "shape.h"
#include <math.h>
#include "functions.h"
//#include <cblas.h>

/*
func::func()
{
}

func::func(int v_iT)
{
	m_v11 = (float *)malloc(v_iT*sizeof(float));
	m_iT = v_iT;
}
*/







shape::shape()
{

		
}

shape::shape( int v_iT)
{
	m_pfPhi = (float *)malloc(v_iT*sizeof(float));
	m_pfTheta = (float *)malloc(v_iT*sizeof(float));
	m_v11 = (float *)malloc(v_iT*sizeof(float));
	m_v12 = (float *)malloc(v_iT*sizeof(float));
        m_v13 = (float *)malloc(v_iT*sizeof(float));
	m_iT = v_iT;
      
}

shape::shape(int n, int v_iT)
{
	m_pfPhi = (float *)malloc(v_iT*sizeof(float));
	m_pfTheta = (float *)malloc(v_iT*sizeof(float));
	m_v11 = (float *)malloc(v_iT*sizeof(float));
	m_v12 = (float *)malloc(v_iT*sizeof(float));
    if(n == 3)
        m_v13 = (float *)malloc(v_iT*sizeof(float));
	m_iT = v_iT;
    m_n = n;

    arr = (float **)malloc(n * sizeof(float *));
    for (int i=0; i<n; i++)
        arr[i] = (float *)malloc(v_iT * sizeof(int));

}

shape::shape(shape &s2)
{
	m_pfPhi = (float *)malloc(s2.m_iT*sizeof(float));
	m_pfTheta = (float *)malloc(s2.m_iT*sizeof(float));
		
	m_iT = s2.m_iT;
	if(m_pfPhi != NULL && m_pfTheta != NULL)
	{
		for(int i = 0; i < s2.m_iT; i ++)
		{
			m_pfPhi[i] = s2.m_pfPhi[i];
			m_pfTheta[i] = s2.m_pfTheta[i];
		}	
	}
}
shape::shape(int v_iT,float *v_pfPhi, float *v_pfTheta)
{
	m_pfPhi = (float *)malloc(v_iT*sizeof(float));
	m_pfTheta = (float *)malloc(v_iT*sizeof(float));
	
	
	copy(v_iT,v_pfPhi,m_pfPhi);
	copy(v_iT,v_pfTheta,m_pfTheta);	
	m_iT = v_iT;
}

//Overloading the assignment operator
shape& shape::operator = (const shape &s)
{
	if(this == &s)		// To prevent assignment to itself
		return *this;
	delete m_pfPhi;
	delete m_pfTheta;
	
	m_iT = s.m_iT;
	m_pfPhi = new float[m_iT];
	m_pfTheta = new float[m_iT];
	for(int i = 0; i < m_iT; i ++)
	{
		m_pfPhi[i] = s.m_pfPhi[i];
		m_pfTheta[i] = s.m_pfTheta[i];
	}
	return *this;
}

void shape::create(int v_iT)
{
	m_pfPhi = (float *)calloc(v_iT,sizeof(float));
	m_pfTheta = (float *)calloc(v_iT,sizeof(float));
	m_iT = v_iT;
}

void shape::create(int v_iT,float *v_Phi, float *v_Theta)
{
	m_pfPhi = (float *)calloc(v_iT,sizeof(float));
	m_pfTheta = (float *)calloc(v_iT,sizeof(float));
	copy(v_iT,v_Phi,m_pfPhi);
	copy(v_iT,v_Theta,m_pfTheta);	
	m_iT = v_iT;
}

void shape::Initalize(int v_iT,float *v_Phi, float *v_Theta)  //Assumes already created  shape object
{
	copy(v_iT,v_Phi,m_pfPhi);
	copy(v_iT,v_Theta,m_pfTheta);	
	m_iT = v_iT;
}

 
float shape::L2_Costfn(shape & s2)
{
	int i = 0;
	float tmp_phi = 0;
	float tmp_theta = 0;
	float cost = 0;
	for(i = 0; i < m_iT; i ++)
	{
//		cost = cost + pow(m_pfPhi[i] - s2.m_pfPhi[i],2) + pow(m_pfTheta[i] - s2.m_pfTheta[i],2);
		cost = cost + (m_pfPhi[i] - s2.m_pfPhi[i])*(m_pfPhi[i] - s2.m_pfPhi[i]) + (m_pfTheta[i] - s2.m_pfTheta[i])*(m_pfTheta[i] - s2.m_pfTheta[i]);
	}
	return cost;
}

void shape::copy(int T, float *src, float *dest)
{
	for(int i = 0; i < T; i++)
	{
		dest[i] = src[i];
	}
}


shape::~shape()
{
	free(m_pfPhi);
	free(m_pfTheta);
    for (int i=0; i<m_n; i++)
        free(arr[i]);
    free(arr);

}