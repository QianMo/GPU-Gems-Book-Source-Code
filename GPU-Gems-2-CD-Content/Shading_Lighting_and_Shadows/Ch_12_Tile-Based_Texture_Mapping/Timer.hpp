/*
 * Timer.hpp
 *
 * Li-Yi Wei
 * 4/15/2003
 *
 */

#ifndef _TIMER_HPP
#define _TIMER_HPP

#include <winsock.h>

class Timer
{
public:
    Timer(void);
    virtual ~Timer(void);

    // start the timer
    // return 1 if successful, 0 else
    int Start(void);

    // stop the timer
    // return 1 if successful, 0 else
    int Stop(void);

    // get the real (elapsted) time in seconds
    double ElapsedTime(void) const;

    // get the current system time in seconds
    double CurrentTime(void) const;
    
protected:
    typedef DWORD Time;
    
    // compute the elapsed time in seconds
    double ElapsedTime(const Time & startTime,
                       const Time & endTime) const;

protected:
    Time _startRealTime;
    Time _stopRealTime;
    int _started;
};

#endif
