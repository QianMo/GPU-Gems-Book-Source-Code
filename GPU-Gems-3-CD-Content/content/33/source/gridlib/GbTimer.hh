/*----------------------------------------------------------------------
|
| $Id: GbTimer.hh,v 1.3 2004/11/08 11:01:27 DOMAIN-I15+prkipfer Exp $
|
+---------------------------------------------------------------------*/
#ifndef GBTIMER_HH
#define GBTIMER_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"

#ifdef WIN32
#  include "_windows.h"
#else
#  include <unistd.h>
#  include <sys/time.h>
#endif

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

/*!
  \brief A timer that can report the frequency of an event or monitor CPU cycles
  \author Peter Kipfer
  $Revision: 1.3 $
  $Date: 2004/11/08 11:01:27 $

  \note This is a version modified for GPUGems from the original gridlib file

  This is a simple timer class that can report the frequency at which an
  event occurs per second, if it is notified of each occurring event.
  For example, you can measure the number of frames rendered per second.
*/
class GRIDLIB_API GbTimer
{
public:
    /*!
      \brief Construct the timer
      \param maxFrames Number of events to wait before the average rate per second is computed
    */
    GbTimer(int maxFrames = 100)
        : framecount_(0)
	, fps_(0.0f)
	, updatedTiming_(false)
    {
#ifdef WIN32
        QueryPerformanceFrequency(&frequency_);
#endif
        maxFrames_ = maxFrames;
    }
    
    /*!
      \brief Notify the time of an event

      If you want to measure frames rendered per second you should call this after glSwapBuffers.
    */
    INLINE void frame() {
        
        framecount_++;
        
        if(framecount_ >= maxFrames_) {
#        ifdef _WIN32
            QueryPerformanceCounter(&newCount_);
            fps_ = float(maxFrames_) * float(frequency_.QuadPart) / float(newCount_.QuadPart - oldCount_.QuadPart);
#        else
            timeval tmptime;
            gettimeofday (&tmptime,NULL);
            newCount_ = tmptime.tv_sec+tmptime.tv_usec*.000001;
            
            fps_ = float(maxFrames_) * 1.0f / float(newCount_-oldCount_);
#        endif
            
#        ifdef _WIN32
            QueryPerformanceCounter(&oldCount_);
#        else
            gettimeofday (&tmptime,NULL);
            oldCount_ = tmptime.tv_sec+tmptime.tv_usec*.000001;
#        endif
            framecount_ = 0;
            updatedTiming_ = true;
        }
        else {
            updatedTiming_ = false;
        }
        
    }

	/*!
      \brief Start a stopwatch

	  This method forces to store the current time as starting time for a 
	  stopwatch application. See the stop() method for reading the elapsed time.
	*/
	INLINE void start() {
#        ifdef _WIN32
            QueryPerformanceCounter(&oldCount_);
#        else
            timeval tmptime;
            gettimeofday (&tmptime,NULL);
            oldCount_ = tmptime.tv_sec+tmptime.tv_usec*.000001;
#        endif
            framecount_ = 0;
            updatedTiming_ = false;
	}

	/*!
	 \brief Stop a stopwatch
	 \return Elapsed time in seconds

	 This method reports the elapsed time in seconds since the start() method
	 has been called. It does not update the fps average.
    */
	INLINE float stop() {
            updatedTiming_ = true;
#        ifdef _WIN32
            QueryPerformanceCounter(&newCount_);
            return float(newCount_.QuadPart - oldCount_.QuadPart) / float(frequency_.QuadPart);
#        else
            timeval tmptime;
            gettimeofday (&tmptime,NULL);
            newCount_ = tmptime.tv_sec+tmptime.tv_usec*.000001;
			return float(newCount_-oldCount_);
#        endif
	}

    /*!
      \brief Notify the timer of an event
      \param print_if_update If set, report the current average rate

      The average rate will only be reported if it has been recomputed because
      this call triggered the maximum number of call to average that has been
      set in the constructor.
    */
    INLINE void frame(GbBool print_if_update) {
        frame();
        if(print_if_update) {
            if(updatedTiming_)
                infomsg(fps_ << " fps");
        }
    }

    /*!
      \brief Get the current average rate
      \return The current average rate
    */
    INLINE float getFps() { return fps_; }

    /*!
      \brief Query whether the last event notification triggered a recomputation of the average rate
      \return True, if the last event notification triggered a recomputation of the average rate
    */
    INLINE GbBool timingUpdated() { return updatedTiming_; }

private:
    //! Current number of reported events
    int framecount_;
#ifdef WIN32
    LARGE_INTEGER oldCount_, newCount_;
    LARGE_INTEGER frequency_;
#else
    //! System timer values
    double oldCount_, newCount_;
#endif
    //! Number of events to average
    int maxFrames_;
    //! Current average rate
    float fps_;
    //! Whether the last event notification triggered a recomputation of the average rate
    GbBool updatedTiming_;    
};

#include "Profiler.hh"

//#ifndef OUTLINE
//#include "GbTimer.in"
//#endif

#endif // GBTIMER_HH
