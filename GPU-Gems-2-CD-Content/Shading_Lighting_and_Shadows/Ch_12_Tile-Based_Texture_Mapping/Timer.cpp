/*
 * Timer.cpp
 *
 * Li-Yi Wei
 * 4/15/2003
 *
 */

#include <windows.h>
#include <mmsystem.h>

#include "Timer.hpp"

Timer::Timer(void) : _started(0)
{
    // nothing to do
}

Timer::~Timer(void)
{
    // nothing to do
}

int Timer::Start(void)
{
    if(timeBeginPeriod(1) == TIMERR_NOERROR)
    {
        _started = 1;
        _startRealTime = timeGetTime();
        return 1;
    }
    else
    {
        return 0;
    }
}

int Timer::Stop(void)
{
    _started = 0;
    _stopRealTime = timeGetTime();
    timeEndPeriod(1);

    return 1;
}

double Timer::ElapsedTime(const Time & startTime,
			  const Time & endTime) const
{
    return (_stopRealTime - _startRealTime)*0.001;
}

double Timer::ElapsedTime(void) const
{
    return ElapsedTime(_startRealTime, _stopRealTime);
}

double Timer::CurrentTime(void) const
{
    if(_started)
    {
        return 0;
    }
    else
    {
        if(timeBeginPeriod(1) == TIMERR_NOERROR)
        {
            Time value = timeGetTime();
            timeEndPeriod(1);
            return value * 0.001;
        }
        else
        {
            return 0;
        }
    }
}
