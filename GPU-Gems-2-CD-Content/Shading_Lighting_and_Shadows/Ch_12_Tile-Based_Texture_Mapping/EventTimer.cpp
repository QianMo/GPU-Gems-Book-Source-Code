/*
 * EventTimer.cpp
 *
 * Li-Yi Wei
 * 9/23/2003
 *
 */

#include <iostream>
using namespace std;

#include "EventTimer.hpp"

const float EventTimer::_ILLEGAL_VALUE = -1;
 
EventTimer::EventTimer(const int numRecordableEvents) : _eventTable(numRecordableEvents), _eventPointer(0)
{
    Clear();
}

EventTimer::~EventTimer(void)
{
    // nothing to do
}

void EventTimer::Clear(void)
{
    for(unsigned int i = 0; i < _eventTable.size(); i++)
    {
        _eventTable[i] = _ILLEGAL_VALUE;
    }

    _eventPointer = 0;
}

int EventTimer::RecordTime(const float eventTime)
{
    _eventTable[_eventPointer] = eventTime;

    _eventPointer = (_eventPointer + 1)%_eventTable.size();

    return 1;
}

float EventTimer::ElapsedTime(const int inputWhichEvent,
                              const int lookBackWindowSize) const
{
    float result = -1;

    // toroidally map whichEvent into the input range
    int whichEvent = inputWhichEvent;
    if(whichEvent < 0)
    {
        // use most recent event
        whichEvent = \
            (_eventPointer - 1 + _eventTable.size())%_eventTable.size();
    }
    else
    {
        whichEvent = whichEvent%_eventTable.size();
    }
    
    if((lookBackWindowSize >= 0) && (lookBackWindowSize < _eventTable.size()))
    {
        int previousEvent =
            (whichEvent - lookBackWindowSize +
             _eventTable.size())%_eventTable.size();

        if( (_eventTable[whichEvent] != _ILLEGAL_VALUE) &&
            (_eventTable[previousEvent] != _ILLEGAL_VALUE) )
        {
            result = _eventTable[whichEvent] - _eventTable[previousEvent];
        }
    }

    // done
    return result;
}
