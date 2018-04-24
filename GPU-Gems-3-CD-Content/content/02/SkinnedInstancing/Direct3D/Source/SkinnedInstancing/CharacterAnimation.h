//----------------------------------------------------------------------------------
// File:   CharacterAnimation.h
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

#pragma once 

#include <vector>

class CharacterSkeleton;
struct CharacterBone;

/*
    A single animation for a character.  Hold a list of interpolated frames.
*/
class CharacterAnimation
{
public:
    CharacterAnimation();
    ~CharacterAnimation();

    typedef std::vector<D3DXMATRIX *> FrameList;

    D3DXMATRIX *GetFrameAt(float time);
    D3DXMATRIX *GetEncodedFrameAt(float time);
    D3DXMATRIX *GetFrame(int index);
    D3DXMATRIX *GetEncodedFrame(int index);
    int GetFrameIndexAt(float time);
    int GetNumFrames(){return (int)frames.size();}

    float duration;
    float timeStep;

    std::string name;
    
    FrameList frames;
    FrameList encodedFrames;
};