#include "timing.h"
#include <iostream>
#include <string>
#include <map>
int64 timerRes;
int64 stop;
int64 start;
int tallymax (int a, int b) {
  return a>b?a:b;
}
brook::stream dummystr;
FILE * streamlog=0;
FILE * streamsummary=0;
typedef std::map <std::string,int64> kstype;
kstype kernelsummary;

void WriteSummary() {
  for (kstype::iterator i = kernelsummary.begin();i!=kernelsummary.end();++i) {
    char siz[128]={0};
    char * here=&siz[127];
    int64 num=(*i).second;
    for(unsigned int j=0;j<127;++j) {      

      if (num||j==0) {
        here--;
        *here='0'+(char)(num%10);
      }
      else
        break;
      num/=10;
    }
    fprintf (streamsummary,"%s %s\n",(*i).first.c_str(), here);

  }
}
void TallyKernel(const char * name,
                 const brook::StreamInterface *a,
                 const brook::StreamInterface *b,
                 const brook::StreamInterface *c,
                 const brook::StreamInterface *d) {
  unsigned int maxwid=a->getExtents()[0];
  unsigned int maxhei=a->getExtents()[1];
  if (b!=(brook::StreamInterface*)dummystr) {
    maxwid = tallymax(b->getExtents()[0],maxwid);
    maxhei = tallymax(b->getExtents()[1],maxhei);
  }
  if (c!=(brook:: StreamInterface*)dummystr) {
    maxwid = tallymax(c->getExtents()[0],maxwid);
    maxhei = tallymax(c->getExtents()[1],maxhei);
  }
  if (d!=(brook::StreamInterface*)dummystr) {
    maxwid = tallymax(d->getExtents()[0],maxwid);
    maxhei = tallymax(d->getExtents()[1],maxhei);
  }
  kernelsummary[name]+=maxwid*maxhei;
  if (streamlog) {
    fprintf (streamlog,"%s %d %d\n",name,maxwid,maxhei);
  }
}
               

void GenSetupMillisTimer(int argc, char**argv) {
  if (streamlog) fclose(streamlog);
  if (streamsummary) {
    WriteSummary();
    fclose(streamsummary);
  }
  std::string filename("log");
  int i;
  for ( i=1;i<argc;++i){
    if (argv[i][0]>='0'&&argv[i][1]<='9')
      filename+="-";
    filename+=argv[i];
  }
  streamlog = fopen (filename.c_str(),"w");
  filename="sum";
  for ( i=1;i<argc;++i){
    if (argv[i][0]>='0'&&argv[i][1]<='9')
      filename+="-";
    filename+=argv[i];
  }
  streamsummary = fopen (filename.c_str(),"w");
}

#ifdef _WIN32
int64
GetTime(void)
{
   static double cycles_per_usec;
   LARGE_INTEGER counter;

   if (cycles_per_usec == 0) {
      static LARGE_INTEGER lFreq;
      if (!QueryPerformanceFrequency(&lFreq)) {
         std::cerr << "Unable to read the performance counter frquency!\n";
         return 0;
      }

      cycles_per_usec = 1000000 / ((double) lFreq.QuadPart);
   }

   if (!QueryPerformanceCounter(&counter)) {
      std::cerr << "Unable to read the performance counter!\n";
      return 0;
   }

   return ((int64) (((double) counter.QuadPart) * cycles_per_usec));
}

// Tim is evil...
#pragma comment(lib,"winmm")

unsigned int GetTimeMillis(void) {
  return (unsigned int)timeGetTime();
}

//  By default in 2000/XP, the timeGetTime call is set to some resolution
// between 10-15 ms query for the range of value periods and then set timer
// to the lowest possible.  Note: MUST make call to corresponding
// CleanupMillisTimer
void SetupMillisTimer(int argc, char* *argv) {

  TIMECAPS timeCaps;
  timeGetDevCaps(&timeCaps, sizeof(TIMECAPS)); 

  if (timeBeginPeriod(timeCaps.wPeriodMin) == TIMERR_NOCANDO) {
    std::cerr << "WARNING: Cannot set timer precision.  Not sure what precision we're getting!\n";
  }
  else {
    timerRes = timeCaps.wPeriodMin;
    //std::cout << "(* Set timer resolution to " << timeCaps.wPeriodMin << " ms. *)\n";
  }
  GenSetupMillisTimer(argc,argv);
}
void CleanupMillisTimer(void) {
  if ((int64)timeEndPeriod((unsigned int)timerRes) == (int64)TIMERR_NOCANDO) {
    std::cerr << "WARNING: bad return value of call to timeEndPeriod.\n";
  }
  if (streamlog)
    fclose(streamlog);
  if (streamsummary) {
    WriteSummary();
    fclose(streamsummary);
  }
  streamlog=0;
  streamsummary=0;
}
#else
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
void SetupMillisTimer(int argc, char**argv) {
  GenSetupMillisTimer(argc,argv);
}
void CleanupMillisTimer(void) {
  if (streamlog) fclose (streamlog);
  if (streamsummary) {
    WriteSummary();
    fclose(streamsummary);
  }

  streamlog=0;
}
int64 GetTime (void) {
  struct timeval tv;
  timerRes = 1000;
  gettimeofday(&tv,NULL);
  int64 temp = tv.tv_usec;
  temp+=tv.tv_sec*1000000;
  return temp;
}
unsigned int GetTimeMillis () {
  return (unsigned int)(GetTime ()/1000);
}
#endif


