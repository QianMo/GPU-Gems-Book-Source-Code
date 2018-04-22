///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2003, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
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
///////////////////////////////////////////////////////////////////////////

#include <FrameRateCounter.h>

#include <sys/time.h>
#include <time.h>

namespace ExrPlay
{

namespace
{

double
now ()
{
    struct timeval tv;
    gettimeofday (&tv, 0);
    return tv.tv_sec + (double) tv.tv_usec / 1.0e6;
}

} // namespace


FrameRateCounter::FrameRateCounter (int avePeriod)
    : _head (0),
      _totalBytes (0)
{
    avePeriod = avePeriod < 2 ? 2 : avePeriod;

    double t = now ();

    _then.resize (avePeriod, t);
    _size.resize (avePeriod, 0);
}


int
FrameRateCounter::tail () const throw ()
{
    return (_head + 1) % _then.size ();
}


float
FrameRateCounter::tick (size_t nbytes) throw ()
{
    double t = now ();
    double fps = 1. / (t - _then[_head]);

    _totalBytes = _totalBytes + nbytes - _size[tail ()];

    _head = (_head + 1) % _then.size ();

    _then[_head] = t;
    _size[_head] = nbytes;

    return fps;
}


float
FrameRateCounter::fps () const throw ()
{
    return (_then.size () - 1) / (_then[_head] - _then[tail ()]);
}


float
FrameRateCounter::bandwidth () const throw ()
{
    return (_totalBytes - _size[tail ()]) / (_then[_head] - _then[tail ()]);
}

} // namespace ExrPlay
