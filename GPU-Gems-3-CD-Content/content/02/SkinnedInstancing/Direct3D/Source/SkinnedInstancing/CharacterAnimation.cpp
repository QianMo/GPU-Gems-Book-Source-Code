//----------------------------------------------------------------------------------
// File:   CharacterAnimation.cpp
// Author: Bryan Dudash
// Email:  sdkfeedback@nvidia.com
// 
// Copyright (c) 2007 NVIDIA Corporation. All rights reserved.
//
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA OR ITS SUPPLIERS
// BE  LIABLE  FOR  ANY  SPECIAL,  INCIDENTAL,  INDIRECT,  OR  CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
//
//
//----------------------------------------------------------------------------------

#include "DXUT.h"
#include "CharacterAnimation.h"

CharacterAnimation::CharacterAnimation()
{
    timeStep = 1/30.f;    // $$ 30fps animation interpolation off the curves
}

CharacterAnimation::~CharacterAnimation()
{
    while(frames.size())
    {
        delete [] frames.back();
        frames.pop_back();
    }

    while(encodedFrames.size())
    {
        delete [] encodedFrames.back();
        encodedFrames.pop_back();
    }
}

D3DXMATRIX *CharacterAnimation::GetFrameAt(float time)
{
    return frames[GetFrameIndexAt(time)];
}

D3DXMATRIX *CharacterAnimation::GetEncodedFrameAt(float time)
{
    return encodedFrames[GetFrameIndexAt(time)];
}

D3DXMATRIX *CharacterAnimation::GetFrame(int index)
{

    assert(index < (int)frames.size());
    return frames[index];
}

D3DXMATRIX *CharacterAnimation::GetEncodedFrame(int index)
{
    assert(index < (int)encodedFrames.size());
    return encodedFrames[index];

}

int CharacterAnimation::GetFrameIndexAt(float time)
{
    // get a [0.f ... 1.f) value by allowing the percent to wrap around 1
    float percent = time / duration;
    int percentINT = (int)percent;
    percent = percent - (float)percentINT;

    return (int)((float)frames.size() * percent);
}
