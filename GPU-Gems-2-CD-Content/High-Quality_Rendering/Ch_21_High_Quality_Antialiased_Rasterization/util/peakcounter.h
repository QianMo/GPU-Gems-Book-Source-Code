/////////////////////////////////////////////////////////////////////////////
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// (This is the Modified BSD License)
/////////////////////////////////////////////////////////////////////////////


#ifndef PEAKCOUNTER_H
#define PEAKCOUNTER_H


// PeakCounter is a simple counter that only supports += and -=, and
// keeps track of the peak value and total increments requested, in
// addition to the current count.
//
// The main use for this is to keep track of memory usage.  Each
// category of dynamically-allocated item should have a static
// PeakCounter, which has a += for every new or malloc, and a -= for
// every delete or free.
// 
// Note that PeakCounter itself is not thread-safe -- you need to make
// sure that it is modified only inside thread-safe code.
//
// To use a PeakCounter m, just do 
//    m += size;
// whenever allocating size (bytes or number of items, your choice), and
//    m -= size;
// whenever freeing size.  This makes for easy stats printing, like this:
//    printf ("Foo allocation: %lld requested, %d current, %d peak (%d KB)\n",
//            m.requested(), m.current(), m.peak(),
//            (m.peak()*sizeof(item))/1024);
// (note the %lld for the long long requested)
//



// Some includes we always need
#include <iostream>
#include <cassert>



class PeakCounter {

private:
    long long _requested;  // Sum of all requests
    long _current;         // How many currently allocated?
    long _peak;            // Peak number allocated

public:
    PeakCounter () : _requested(0), _current(0), _peak(0) { }

    void reset (void) {_requested = 0; _current = 0; _peak = 0; }

    const PeakCounter& operator+= (long size) {
	assert (size >= 0);
	// += always increases requested & current; peak increases if necessary
	_requested += size;
	_current += size;
	if (_current > _peak)
	    _peak = _current;
	return *this;
    }

    const PeakCounter& operator-= (long size) {
	assert (size >= 0);
	// -= only decreases current; doesn't affect peak or requested
	_current -= size;
	assert (_current >= 0);
	return *this;
    }

    // Note:  The peak is incorrect after this operation!
    const PeakCounter& operator+= (const PeakCounter& rhs) {
        _requested += rhs._requested;
        _current += rhs._current;
        _peak += rhs._peak;
        return *this;
    }
    
    const PeakCounter& operator++ ()    { *this += 1;  return *this; }
    const PeakCounter& operator++ (int) { *this += 1;  return *this; }
    const PeakCounter& operator-- ()    { *this -= 1;  return *this; }
    const PeakCounter& operator-- (int) { *this -= 1;  return *this; }

    long long requested (void) const { return _requested; }
    long current (void) const { return _current; }
    long peak (void) const { return _peak; }
};



// AverageCounter stores a running average of a counter.  The denominator
// of the average is the number of times that the counter has changed.
// This may not be what you are expecting since it is not an average over
// time or another well-defined value.
//
// The primary use for this counter is to keep track of the average size
// of some resource, memory, queue lengths, or so forth.  Each time the
// value is changed, the average is recomputed so you can have a sense of
// the average value in addition to the total and peak counts.
//
// Note that like the PeakCounter, the AverageCounter is not thread-safe.
//
// The AverageCounter can be used just like a PeakCounter

class AverageCounter {
    
 private:
    long long _count;
    double _average;
    PeakCounter peakcounter;
    
    void update_average (long delta) {
        _average = (_average * _count + (current() + delta)) / (_count + 1);
        ++_count;
    }
    
 public:
    AverageCounter () : _count(0), _average(0) { }
    
    void reset (void) { _count = 0; _average = 0; peakcounter.reset(); }

    const AverageCounter &operator+= (long delta) {
        update_average (delta);
        peakcounter += delta;
        return *this;
    }
    
    const AverageCounter &operator-= (long delta) {
        update_average (-delta);
        peakcounter -= delta;
        return *this;
    }

    const AverageCounter &operator+= (const AverageCounter &rhs) {
        peakcounter += rhs.peakcounter;
        _average = (_average * _count + rhs._average * rhs._count) /
            (_count + rhs._count);
        _count += rhs._count;
        return *this;
    }

    const AverageCounter& operator++ ()    { *this += 1;  return *this; }
    const AverageCounter& operator++ (int) { *this += 1;  return *this; }
    const AverageCounter& operator-- ()    { *this -= 1;  return *this; }
    const AverageCounter& operator-- (int) { *this -= 1;  return *this; }

    long long requested (void) const { return peakcounter.requested(); }
    long current (void) const { return peakcounter.current(); }
    long peak (void) const { return peakcounter.peak(); }
    double average (void) const { return _average; }
    long long count (void) const { return _count; }
};



#endif /* !defined(PEAKCOUNTER_H) */
