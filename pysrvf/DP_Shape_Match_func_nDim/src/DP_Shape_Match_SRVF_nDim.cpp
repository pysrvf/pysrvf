#include <stdio.h>
#include "shape.h"
#include <time.h>
#include <stdlib.h>
#include "functions.h"
//#include "params.h"
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
	clock_t start, finish;
	float duration;
  	//Declaration of variables
	int vl_iNum_Shapes = 0;
	float a = 1.0; 
	float b = 1.0;
	shape *s1 = NULL; shape *s2 = NULL;
//	params *v_pparams = NULL;
	float **Energy = NULL;
	float *gamma_hat = NULL;
        int i = 0;
        
	FILE *vl_fpgamma = NULL;
	int vl_ishift = 0;
	if(argc != 3)
	{
		printf("\nError: Incorrect number of arguments\n");
		printf("Usage: %s <q function pair> <output match> \n \n",argv[0]);
		exit(1);
	}

	if(Read_q1_q2(argv[1],&s1,&s2))
	{

        start = clock();
        gamma_hat = (float *)malloc(s1->m_iT * sizeof(float));
        MatchQ(s1, s2,gamma_hat, Energy);
        finish = clock();
        duration = (float)(finish - start) / CLOCKS_PER_SEC;
//        printf("duration: %f",duration);
        vl_fpgamma = fopen(argv[2],"wb");
        WriteGammaV(vl_fpgamma,gamma_hat,vl_ishift,s2);
        fclose(vl_fpgamma);

	}

/*	if(Read_v1_v2(argv[1],&s1,&s2,&a,&b) == true)	
	{
//            v_pparams = new params(s1->m_iT, 3,a, b, 2);
            start = clock();
            gamma_hat = (float *)malloc(s1->m_iT * sizeof(float));

            MatchQ(s1, s2,gamma_hat, Energy);
            finish = clock();
            duration = (float)(finish - start) / CLOCKS_PER_SEC;
//             printf("duration: %f",duration);
            vl_fpgamma = fopen(argv[2],"wb");	
            WriteGammaV(vl_fpgamma,gamma_hat,vl_ishift,s2);
            fclose(vl_fpgamma);
//            free(v_pparams);
	}
	*/
	return 0;
}
