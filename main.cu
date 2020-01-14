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

#define DEBUG 0

static double get_second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

template <typename T, typename T_REAL> 
void gtsv_spike_partial_diag_pivot(const T* dl, const T* d, const T* du, T* b,const int m,const int k);
template <typename T> 
void dtsvb_spike_v1(const T* dl, const T* d, const T* du, T* b,const int m);


//utility
#define EPS 1e-20


template <typename T> 
void mv_test
(
	T* x,
	const T* a,
	const T* b,
	const T* c,
	const T* d,
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



template <typename T, typename T_REAL> 
void compare_result
(
	const T *x,
	const T *y,
	const int sys_num,
	const int len,
	const int rhs,
	const T_REAL abs_err,
	const T_REAL re_err,
	const int p_bound,
	const int k_bound,
	const int tx  //for debug

)
{
	int m = len*sys_num;
	
	T_REAL err = 0.0;
	T_REAL sum_err=0.0;
	T_REAL total_sum=0.0;
	T_REAL r_err=1.0;
	T_REAL x_2=0.0;
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
				T diff = cuSub(x[k*m+j*len+i], y[k*m+j*len+i]);
				err = cuReal( cuMul(diff, cuConj(diff) ));
				sum_err +=err;
				x_2= cuReal( cuMul(x[k*m+j*len+i], cuConj(x[k*m+j*len+i])));
				total_sum+=x_2;
				
				//avoid overflow in error check
				r_err = x_2>EPS?err/x_2:0.0;
				if(err >abs_err || r_err>re_err)
				{
					if(p<p_bound)
					{
						printf("!!!!!!ERROR happen at system %d element %d  cpu %lf and gpu %lf at %d\n",j,i, cuReal(x[k*m+j*len+i]),cuReal(y[k*m+j*len+i]),i%tx);
						printf("!!!!!!its abs err us %le and rel err is %le\n",err,r_err);
					}
					p++;
				}
				
				if(t<16 )
				{
					printf("check happen at system %d element %d  cpu %lf and gpu %lf\n",j,i,cuReal(x[k*m+j*len+i]),cuReal(y[k*m+j*len+i]));
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

//This is a testing gtsv function
template <typename T, typename T_REAL> 
void test_gtsv_v1(int m)
{
	double start,stop;
	T *h_dl;
	T *h_d;
	T *h_du;
	T *h_b;
	
	T *h_x_gpu;
	T *h_b_back;

	T *dl;
	T *d;
	T *du;
	T *b;

	
	//allocation
	{
		h_dl=(T *)malloc(sizeof(T)*m);
		h_du=(T *)malloc(sizeof(T)*m);
		h_d=(T *)malloc(sizeof(T)*m);
		h_b=(T *)malloc(sizeof(T)*m);
		
		h_x_gpu=(T *)malloc(sizeof(T)*m);
		h_b_back=(T *)malloc(sizeof(T)*m);
				
		cudaMalloc((void **)&dl, sizeof(T)*m); 
		cudaMalloc((void **)&du, sizeof(T)*m); 
		cudaMalloc((void **)&d, sizeof(T)*m); 
		cudaMalloc((void **)&b, sizeof(T)*m); 

		cudaMemset(d, 0, m * sizeof(T));
		cudaMemset(dl, 0, m * sizeof(T));
		cudaMemset(du, 0, m * sizeof(T));
	}
	

	
	int k;
	srand(54321);
	//generate random data
	h_dl[0]   = cuGet<T>(0);
	h_d[0]    = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0 );
	h_du[0]   = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
	h_dl[m-1] = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
	h_d[m-1]  = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
	h_du[m-1] = cuGet<T>(0);
	
	h_b[0]    = cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0 );
	h_b[m-1]  =cuGet<T>(  (rand()/(double)RAND_MAX)*2.0-1.0 );
	
	for(k=1;k<m-1;k++)
	{
		h_dl[k] =cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
		h_du[k] =cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
		h_d[k]  =cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
		h_b[k]  =cuGet<T>( (rand()/(double)RAND_MAX)*2.0-1.0);
	}
	
	
   //Memory copy
	cudaMemcpy(dl, h_dl, m*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d, h_d, m*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(du, h_du, m*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(b, h_b, m*sizeof(T), cudaMemcpyHostToDevice);

	//this is for general matrix
    start = get_second();
    gtsv_spike_partial_diag_pivot<T,T_REAL>( dl, d, du, b,m,1);
    cudaDeviceSynchronize();
	stop = get_second();
    printf("test_gtsv_v1 m=%d time=%.6f\n", m, stop-start);    

  	//copy back 
	cudaMemcpy(h_x_gpu, b, m*sizeof(T), cudaMemcpyDeviceToHost);

    mv_test<T>(h_b_back,h_dl,h_d,h_du,h_x_gpu,1,m,1);
    //backward error check
	int b_dim=128;  //for debug
	compare_result<T,T_REAL>(h_b,h_b_back,1,m,1,1e-10,1e-10,50,3,b_dim);
}






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
	test_gtsv_v1<double,double>(m);	
    printf("END double test_gtsv testing\n");
	
	printf("double test_gtsv multiple rhs testing\n");
	test_gtsv_v_few<double,double>(m,k);
    printf("END double test_gtsv multiple rhs testing\n");
	
	/*
    printf("Double complex test_gtsv 5 rhs testing\n");    
	test_gtsv_v_few<cuDoubleComplex,double>(m,5);    
    printf("END Double complex test_gtsv 5 rhs\n");
    
	
	
	printf("double test_dtsvb_v1 testing\n");
	test_dtsvb_v1<double,double>(m);
	
	printf("double complex test_dtsvb_v1 testing\n");
	test_dtsvb_v1<cuDoubleComplex,double>(m);
	
	*/	
	//printf("float testing\n");
	//test_dtsvb_v1<float>(m);
  
	return 0;

}
