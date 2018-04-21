#ifndef __TIMER_H_INCLUDED__
#define __TIMER_H_INCLUDED__

#ifdef WIN32
#include <windows.h>
struct Timer  
{
	__int64 start;
	__int64 now;
	__int64 freq;
	double res;
	double e;
 	__forceinline double fps() {
		// return 1/elapsed_time for the frame rate
		QueryPerformanceCounter((LARGE_INTEGER*)&now);
		e = 1./((double)((now - start)*res));
		QueryPerformanceCounter((LARGE_INTEGER*)&start);
		return e;
	}
	__forceinline double elapsed_ms() { // in ms
		QueryPerformanceCounter((LARGE_INTEGER*)&now);
		return 1000.*(double)((now - start)*res);
	}
	__forceinline double elapsed_s() { // in s
		QueryPerformanceCounter((LARGE_INTEGER*)&now);
		return (double)((now - start)*res);
	}
	__forceinline double time() { // in s
		QueryPerformanceCounter((LARGE_INTEGER*)&now);
		return (double)(now*res);
	}
	__forceinline void init() { 
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		res = (double)(1./(double)freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&start);
	}
	__forceinline void reset() { 
		QueryPerformanceCounter((LARGE_INTEGER*)&start);
	}
};
#else
#include <sys/time.h>

#define __forceinline inline

struct Timer  
{
        double uinv;
        struct timeval start;
 	__forceinline double fps() {
		// return 1/elapsed_time for the frame rate
	    struct timeval now;
		gettimeofday(&now, 0);
		long sec = now.tv_sec - start.tv_sec;
		long usec = now.tv_usec - start.tv_usec;
		double e = 1./(sec + usec*uinv);
		start = now;
		return e;
	}
	__forceinline double elapsed_ms() { // in ms
	    struct timeval now;
		gettimeofday(&now, 0);
		long sec = now.tv_sec - start.tv_sec;
		long usec = now.tv_usec - start.tv_usec;
		double e = 1000. * (sec + usec*uinv);
		return e;
	}
	__forceinline double elapsed_s() { // in s
	        struct timeval now;
		gettimeofday(&now, 0);
		long sec = now.tv_sec - start.tv_sec;
		long usec = now.tv_usec - start.tv_usec;
		double e = sec + usec*uinv;
		return e;
	}
	__forceinline double time() { // in s
	    struct timeval now;
		gettimeofday(&now, 0);
		double e = now.tv_sec + now.tv_usec*uinv;
		return e;
	}
	__forceinline void init() { 
		uinv = 1/1000000.;
	    gettimeofday(&start, 0);
	}
	__forceinline void reset() { 
	    init();
	}
};

#endif
#endif
