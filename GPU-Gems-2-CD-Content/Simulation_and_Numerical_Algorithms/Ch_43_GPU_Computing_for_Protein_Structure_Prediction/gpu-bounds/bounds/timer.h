/////////////////////////////////////////////////////////////
//
// programmer: Paulius Micikevicius, pmicikev@cs.ucf.edu
//
// The class CMilliTimer allows timing with milli-second precision
// The two methods, StartTimer() and GetET() start the timer and
// get the elapsed time as a float (in seconds, fraction specifies
// milliseconds over the second, for exaple 1.5 means a second and
// a half), respectively
//

#ifndef __TIMER
	#define	__TIMER

	#ifdef WIN32	// win32 version

		#include <sys/types.h>
		#include <sys/timeb.h>

		class CTimer
		{
		protected:
			_timeb start_time;
		public:
			void Start();
			float GetET();
		};

		void CTimer::Start()
		{
			_ftime(&start_time);
		}

		float CTimer::GetET()
		{
			_timeb current_time;
			_ftime(&current_time);
			float et=(current_time.time+current_time.millitm*0.001)-(start_time.time+start_time.millitm*0.001);
			
			return et;
		}

	#else	// Unix version

		#include <unistd.h>
		#include <sys/time.h>
		#include <time.h>
		#include <fstream.h>

		class CTimer
		{
		protected:
			timeval start,end;
		public:
			void Start();
			double GetET();
		};

		void CTimer::Start()
		{
			gettimeofday(&start, NULL);
		}

		double CTimer::GetET()
		{
			gettimeofday(&end,NULL);
			double
			et=(end.tv_sec+end.tv_usec*0.000001)-(start.tv_sec+start.tv_usec*0.000001);
			return et;
		}

	#endif

#endif

/* sample use

void main()
{
	int x;
	CTimer timer;
	timer.Start();
	for(int i=0;i<32000;i++)
		for(int j=0;j<10000;j++)
			x++;
	double et=timer.GetET();
	cout<<et<<endl;
}
*/