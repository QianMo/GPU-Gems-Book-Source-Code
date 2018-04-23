/*
 * EventTimer.hpp
 *
 * Li-Yi Wei
 * 9/23/2003
 *
 */

#ifndef _EVENT_TIMER_HPP
#define _EVENT_TIMER_HPP

#include <vector>
using namespace std;

class EventTimer
{
public:
    // numRecordableEvents: only record this number of most recent events
    EventTimer(const int numRecordableEvents);

    ~EventTimer(void);

    // clear all event entris
    void Clear(void);
    
    // record the current event time
    int RecordTime(const float eventTime);

    // report the elapsed time for whichEvent
    // with specified look-back window size
    // for example, if whichEvent = 100 and lookBackWindowSize = 10
    // then this function returns elapsed time between event 91 to 100
    // if whichEvent < 0, use the most recent event
    // return something < 0 if input arguments are invalid
    float ElapsedTime(const int whichEvent,
                      const int lookBackWindowSize) const;
    
protected:
    vector<float> _eventTable;
    int _eventPointer;

    static const float _ILLEGAL_VALUE;
};

#endif
