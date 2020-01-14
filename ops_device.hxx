/* Copyright (c) 2009-2013,  NVIDIA CORPORATION

   All rights reserved.
   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the distribution.
   Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used to endorse
   or promote products derived from this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/   
 
#ifndef _CUSPARSE_OPS_HXX_
#define _CUSPARSE_OPS_HXX_

/* Multiplication */


static __inline__ __device__ __host__  double cuMul( double x , double y )
{
    return(x * y);
}



/* Negation */


static __inline__ __device__ __host__  double cuNeg( double x )
{
    return(-x);
}

/* Addition */


static __inline__ __device__ __host__  double cuAdd( double x , double y )
{
    return(x + y);
}

 
/* Subtraction */


static __inline__ __device__ __host__  double cuSub( double x , double y )
{
    return(x - y);
}

  
/* Division */


static __inline__ __device__ __host__  double cuDiv( double x , double y )
{
    return (x / y);
}


/* Fma */


static __inline__ __device__ __host__  double cuFma( double x , double y, double d )
{
    return ((x * y) + d);
}

  

/* absolute value */


static __inline__ __device__ __host__  double cuAbs( double x )
{
    return (fabs(x));
}

 __inline__ __device__ __host__  double cuGet(int x)
{
    return double(x);

}

template <typename T_ELEM> struct __dynamic_shmem__{
    __device__ T_ELEM * getPtr() { 
        extern __device__ void error(void);
        error();
        return NULL;
    }
}; 
/* specialization of the above structure for the desired types */
template <> struct __dynamic_shmem__<float>{
    __device__ float * getPtr() { 
        extern __shared__ float Sptr[];
        return Sptr;
    }
};
template <> struct __dynamic_shmem__<double>{
    __device__ double * getPtr() { 
        extern __shared__ double Dptr[];
        return Dptr;
    }
};

template <> struct __dynamic_shmem__<cuComplex>{
    __device__ cuComplex * getPtr() { 
        extern __shared__ cuComplex Cptr[];
        return Cptr;
    }
};
template <> struct __dynamic_shmem__<cuDoubleComplex>{
    __device__ cuDoubleComplex * getPtr() { 
        extern __shared__ cuDoubleComplex Zptr[];
        return Zptr;
    }
};


#endif
