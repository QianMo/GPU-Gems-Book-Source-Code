#ifndef _FRAME_RATE_COUNTER_H_
#define _FRAME_RATE_COUNTER_H_

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

#include <vector>
#include <sys/types.h>

namespace ExrPlay
{

//-------------------------------------------
//
//    Measure frame rate and bandwidth.
//
//-------------------------------------------

class FrameRateCounter
{
 public:

    //---------------------------------------------------------
    // Constructor.
    //
    // avePeriod is the number of frames over which to average
    // the frame rate and bandwidth statistics.
    //---------------------------------------------------------

    FrameRateCounter (int avePeriod = 24);


    //-----------------------------------------------------------
    // Call this whenever a frame is displayed.  nbytes is
    // the number of bytes in the frame that was just displayed.
    // It returns the instantaneous frame rate.
    //-----------------------------------------------------------

    float tick (size_t nbytes = 0) throw ();


    //-----------------------------------------------------------
    // fps returns the average frame rate.
    //
    // bandwidth returns the average bandwidth.
    //
    // These rates are moving averages over the last whole
    // period's worth of frames.  The rates reported by these
    // two methods are not reliable until tick has been called 
    // once per frame over a whole period.
    //-----------------------------------------------------------

    float fps () const throw ();
    float bandwidth () const throw ();


 private:

    int   tail () const throw ();

    std::vector<double> _then;          // last period's worth of time samples.
    std::vector<size_t> _size;          // last period's worth of frame sizes.
    int                 _head;          // head of ringbuffer.
    size_t              _totalBytes;    // sum of last 24 frame sizes.
};

} // namespace ExrPlay

#endif // _FRAME_RATE_H_
