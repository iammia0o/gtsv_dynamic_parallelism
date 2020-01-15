/*******************************************************************************************************
                              University of Illinois/NCSA Open Source License
                                 Copyright (c) 2012 University of Illinois
                                          All rights reserved.

                                        Developed by: IMPACT Group
                                          University of Illinois
                                      http://impact.crhc.illinois.edu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
  Neither the names of IMPACT Group, University of Illinois, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.

*******************************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "ops_device.hxx"
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <curand_kernel.h>
#include <ctime>

#define DEBUG 0

static __device__ double get_second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

__device__
void gtsv_spike_partial_diag_pivot_v1(const DOUBLE* dl, const DOUBLE* d, const DOUBLE* du, DOUBLE* b,const int m);

// void dtsvb_spike_v1(const DOUBLE* dl, const DOUBLE* d, const DOUBLE* du, DOUBLE* b,const int m);


//utility
#define EPS 1e-20

__device__
void mv_test
(
	DOUBLE* x,
	const DOUBLE* a,
	const DOUBLE* b,
	const DOUBLE* c,
	const DOUBLE* d,
	const int sys_num,
	const int len,
	const int rhs
)
{
	int m=sys_num*len;
	
	for(int j=0;j<rhs;j++)
	{
		x[j*m] =  cuAdd( cuMul(b[0],d[j*m]), cuMul(c[0],d[j*m+1]));
		for(int i=1;i<m-1;i++)
		{	
			//x[i]=  a[i]*d[i-1]+b[i]*d[i]+c[i]*d[i+1];
			x[j*m+i]=  cuMul(a[i],d[j*m+i-1]);
			x[j*m+i]=  cuFma(b[i], d[j*m+i], x[j*m+i]);
			x[j*m+i]=  cuFma(c[i], d[j*m+i+1], x[j*m+i]);
		}
		x[j*m+m-1]= cuAdd( cuMul(a[m-1],d[j*m+m-2]) , cuMul(b[m-1],d[j*m+m-1]) );
	}
}


__device__
void compare_result
(
	const DOUBLE *x,
	const DOUBLE *y,
	const int sys_num,
	const int len,
	const int rhs,
	const DOUBLE abs_err,
	const DOUBLE re_err,
	const int p_bound,
	const int k_bound,
	const int tx
 
)
{
	int m = len*sys_num;
	
	DOUBLE err = 0.0;
	DOUBLE sum_err=0.0;
	DOUBLE total_sum=0.0;
	DOUBLE r_err=1.0;
	DOUBLE x_2=0.0;
	int p=0; //error counter
	int t=0; //check Non counter
	
	for(int k=0;k<rhs;k++)
	{
		if(k<k_bound)
		{
			printf("@@@@@@@@  rhs vector is %d\n",k);
			
		}
		p=0;
		for(int j=0;j<sys_num;j++)
		{
			if(k<k_bound)
				t=0;
			for(int i=0;i<len;i++)
			{
				DOUBLE diff = cuSub(x[k*m+j*len+i], y[k*m+j*len+i]);
				err =  cuMul(diff, diff);
				sum_err +=err;
				x_2= cuMul(x[k*m+j*len+i], x[k*m+j*len+i]);
				total_sum+=x_2;
				
				//avoid overflow in error check
				r_err = x_2>EPS?err/x_2:0.0;
				if(err >abs_err || r_err>re_err)
				{
					if(p<p_bound)
					{
						printf("!!!!!!ERROR happen at system %d element %d  cpu %lf and gpu %lf at %d\n",j,i, (x[k*m+j*len+i]), (y[k*m+j*len+i]),i%tx);
						printf("!!!!!!its abs err us %le and rel err is %le\n",err,r_err);
					}
					p++;
				}
				
				if(t<16 )
				{
					printf("check happen at system %d element %d  cpu %lf and gpu %lf\n",j,i, (x[k*m+j*len+i]), (y[k*m+j*len+i]));
					t++;
				}
			}
		}
		if(k<k_bound)
		{
			if(p==0)
			{
				printf("all correct\n");
			}
			else
			{
				printf("some error\n");
			}
		}
	}
	printf("total abs error is %le\n",sqrt(sum_err));
	printf("total rel error is %le\n",sqrt(sum_err)/sqrt(total_sum));
}


__device__ DOUBLE cudaRand(){
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState state;
	curand_init((unsigned long long)clock() + tId, 0, 0, &state);
	return curand_uniform_double(&state);
}

// This is a testing gtsv function
__global__
void test_gtsv_v1(int m)
{
	double start,stop;
	// DOUBLE *h_dl;
	// DOUBLE *h_d;
	// DOUBLE *h_du;
	// DOUBLE *h_b;
	
	// DOUBLE *h_x_gpu;
	// DOUBLE *h_b_back;

	DOUBLE *dl;
	DOUBLE *d;
	DOUBLE *du;
	DOUBLE *b;

	
	//allocation
	{		
		// h_x_gpu=(DOUBLE *)malloc(sizeof(DOUBLE)*m);
		// h_b_back=(DOUBLE *)malloc(sizeof(DOUBLE)*m);
				
		malloc((void **)&dl, sizeof(DOUBLE)*m); 
		malloc((void **)&du, sizeof(DOUBLE)*m); 
		malloc((void **)&d, sizeof(DOUBLE)*m); 
		malloc((void **)&b, sizeof(DOUBLE)*m); 

		memset(d, 0, m * sizeof(DOUBLE));
		memset(dl, 0, m * sizeof(DOUBLE));
		memset(du, 0, m * sizeof(DOUBLE));
	}
	

	int k;
	// srand(54321);
	//generate random data
	dl[0]   = cuGet(0);
	d[0]    = (cudaRand()/(double)RAND_MAX)*2.0-1.0;
	du[0]   = (cudaRand()/(double)RAND_MAX)*2.0-1.0;
	dl[m-1] = (cudaRand()/(double)RAND_MAX)*2.0-1.0;
	d[m-1]  = (cudaRand()/(double)RAND_MAX)*2.0-1.0;
	du[m-1] = cuGet(0);
	
	b[0]    = (cudaRand()/(double)RAND_MAX)*2.0-1.0 ;
	b[m-1]  =  (cudaRand()/(double)RAND_MAX)*2.0-1.0 ;
	
	for(k=1;k<m-1;k++)
	{
		dl[k] =(cudaRand()/(double)RAND_MAX)*2.0-1.0;
		du[k] =(cudaRand()/(double)RAND_MAX)*2.0-1.0;
		d[k]  =(cudaRand()/(double)RAND_MAX)*2.0-1.0;
		b[k]  =(cudaRand()/(double)RAND_MAX)*2.0-1.0;
	}
	


	//this is for general matrix
    start = get_second();
    gtsv_spike_partial_diag_pivot_v1( dl, d, du, b,m);
    cudaDeviceSynchronize();
	stop = get_second();
    printf("test_gtsv_v1 m=%d time=%.6f\n", m, stop-start);    

  	//copy back 
	// cudaMemcpy(h_x_gpu, b, m*sizeof(DOUBLE), cudaMemcpyDeviceToHost);

 //    mv_test(h_b_back,h_dl,h_d,h_du,h_x_gpu,1,m,1);
 //    //backward error check
	// int b_dim=128;  //for debug
	// compare_result(h_b,h_b_back,1,m,1,1e-10,1e-10,50,3,b_dim);
}



// void test_gtsv_v1(int m)
// {
// 	double start,stop;
// 	DOUBLE *h_dl;
// 	DOUBLE *h_d;
// 	DOUBLE *h_du;
// 	DOUBLE *h_b;
	
// 	DOUBLE *h_x_gpu;
// 	DOUBLE *h_b_back;

// 	DOUBLE *dl;
// 	DOUBLE *d;
// 	DOUBLE *du;
// 	DOUBLE *b;

	
// 	//allocation
// 	{
// 		h_dl=(DOUBLE *)malloc(sizeof(DOUBLE)*m);
// 		h_du=(DOUBLE *)malloc(sizeof(DOUBLE)*m);
// 		h_d=(DOUBLE *)malloc(sizeof(DOUBLE)*m);
// 		h_b=(DOUBLE *)malloc(sizeof(DOUBLE)*m);
		
// 		h_x_gpu=(DOUBLE *)malloc(sizeof(DOUBLE)*m);
// 		h_b_back=(DOUBLE *)malloc(sizeof(DOUBLE)*m);
				
// 		cudaMalloc((void **)&dl, sizeof(DOUBLE)*m); 
// 		cudaMalloc((void **)&du, sizeof(DOUBLE)*m); 
// 		cudaMalloc((void **)&d, sizeof(DOUBLE)*m); 
// 		cudaMalloc((void **)&b, sizeof(DOUBLE)*m); 

// 		cudaMemset(d, 0, m * sizeof(DOUBLE));
// 		cudaMemset(dl, 0, m * sizeof(DOUBLE));
// 		cudaMemset(du, 0, m * sizeof(DOUBLE));
// 	}
	

// 	int k;
// 	srand(54321);
// 	//generate random data
// 	h_dl[0]   = cuGet(0);
// 	h_d[0]    = (rand()/(double)RAND_MAX)*2.0-1.0;
// 	h_du[0]   = (rand()/(double)RAND_MAX)*2.0-1.0;
// 	h_dl[m-1] = (rand()/(double)RAND_MAX)*2.0-1.0;
// 	h_d[m-1]  = (rand()/(double)RAND_MAX)*2.0-1.0;
// 	h_du[m-1] = cuGet(0);
	
// 	h_b[0]    = (rand()/(double)RAND_MAX)*2.0-1.0 ;
// 	h_b[m-1]  =  (rand()/(double)RAND_MAX)*2.0-1.0 ;
	
// 	for(k=1;k<m-1;k++)
// 	{
// 		h_dl[k] =(rand()/(double)RAND_MAX)*2.0-1.0;
// 		h_du[k] =(rand()/(double)RAND_MAX)*2.0-1.0;
// 		h_d[k]  =(rand()/(double)RAND_MAX)*2.0-1.0;
// 		h_b[k]  =(rand()/(double)RAND_MAX)*2.0-1.0;
// 	}
	
	
//    //Memory copy
// 	cudaMemcpy(dl, h_dl, m*sizeof(DOUBLE), cudaMemcpyHostToDevice);
// 	cudaMemcpy(d, h_d, m*sizeof(DOUBLE), cudaMemcpyHostToDevice);
// 	cudaMemcpy(du, h_du, m*sizeof(DOUBLE), cudaMemcpyHostToDevice);
// 	cudaMemcpy(b, h_b, m*sizeof(DOUBLE), cudaMemcpyHostToDevice);

// 	//this is for general matrix
//     start = get_second();
//     gtsv_spike_partial_diag_pivot_v1( dl, d, du, b,m);
//     cudaDeviceSynchronize();
// 	stop = get_second();
//     printf("test_gtsv_v1 m=%d time=%.6f\n", m, stop-start);    

//   	//copy back 
// 	cudaMemcpy(h_x_gpu, b, m*sizeof(DOUBLE), cudaMemcpyDeviceToHost);

//     mv_test(h_b_back,h_dl,h_d,h_du,h_x_gpu,1,m,1);
//     //backward error check
// 	int b_dim=128;  //for debug
// 	compare_result(h_b,h_b_back,1,m,1,1e-10,1e-10,50,3,b_dim);
// }




int main(int argc, char *argv[])
{
      
	int m;
	m = 512*1024+512;
    int k;
	k=2;
	
    if (argc > 1) {
		if(argc>=2)
		{
			m = atoi( argv[1] );
			if(argc>=3)
			{
				k = atoi( argv[2] );
			}
		}
    }
	printf ( "matrix size = %d and rhs is %d \n", m,k);
    
	printf("double test_gtsv testing\n");
	test_gtsv_v1<<<1, 32>>>(m);	
    printf("END double test_gtsv testing\n");

  
	return 0;

}
