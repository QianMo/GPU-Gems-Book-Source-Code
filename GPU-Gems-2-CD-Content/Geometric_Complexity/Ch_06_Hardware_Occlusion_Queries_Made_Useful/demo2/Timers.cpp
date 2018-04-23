#include <time.h>


#ifdef HRTIME
#include <hrtime.h>
#endif

#ifdef _WIN32

#include <windows.h>
#include <sys/types.h>
#include <sys/timeb.h>
#include <time.h>
#include <resource.h>

#else

#include <unistd.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <sys/resource.h>

#endif


#include "Timers.h"

/*real time or user time*/

#define REAL_TIME 


// global file streams
/*ofstream debugs("debug.log");*/

static int hasHRTimer = 0;

#ifdef  _MSC_VER
static LARGE_INTEGER hrFreq;
#endif


#ifdef HRTIME
struct hrtime_struct *hrTimer = NULL;
#endif

void initTiming()
{
#ifdef  _MSC_VER
     hasHRTimer = QueryPerformanceFrequency(&hrFreq);
#else
#ifdef HRTIME
     if (hrtime_is_present()) 
     {
	  hrtime_init();
	  hasHRTimer = (0 == get_hrtime_struct(0, &hrTimer));
	  debugs<<"Using UNIX hires timer"<<endl;
     } else 
     {
	  debugs<<"No UNIX hires timer"<<endl;
     }
#endif
#endif
}

void finishTiming()
{
#ifdef HRTIME
     if (hasHRTimer == 1)
	  free_hrtime_struct(hrTimer);
#endif
  
}


long
getTime()
{
#ifndef  _MSC_VER

#ifdef REAL_TIME

  static struct timeval _tstart;
  static struct timezone tz;

  gettimeofday(&_tstart,&tz);
  return (long)(1000000*_tstart.tv_sec + _tstart.tv_usec);

#else
  
  if (hasHRTimer == 0) {

    static struct rusage r;
    getrusage(RUSAGE_SELF,&r);
    return r.ru_utime.tv_usec+1000000*r.ru_utime.tv_sec;
  } else {
#ifdef HRTIME
    hrtime_t dest;
    get_hrutime(hrTimer, &dest);
    return (long) (dest/500.0);
#else
    return 0;
#endif
  }
#endif
  
#else
  
  if (hasHRTimer == 1) {
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    // return in usec
    return (long) (1000000*counter.QuadPart/(hrFreq.QuadPart));
  } else {
    static struct _timeb mtime;
    _ftime(&mtime);
    
    return 1000*(1000*mtime.time + mtime.millitm);
  }
#endif
}


// return time diff. in ms
double
timeDiff(long time1,long time2) // in ms
{
  const double clk=1.0E-3; // ticks per second
  long t=time2-time1;
  
  return ((t<0)?-t:t)*clk;
}

char
*timeString()
{
  time_t t;
  time(&t);
  return ctime(&t);
}
