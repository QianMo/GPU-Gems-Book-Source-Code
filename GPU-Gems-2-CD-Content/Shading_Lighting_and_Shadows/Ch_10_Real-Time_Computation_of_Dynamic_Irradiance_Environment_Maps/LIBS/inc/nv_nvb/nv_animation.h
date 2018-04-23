/*********************************************************************NVMH1****
File:
nv_animation.h

Copyright (C) 1999, 2002 NVIDIA Corporation
This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.

Comments:


******************************************************************************/

#ifndef _nv_animation_h_ 
#define _nv_animation_h_


//
// Forward declarations
//

class nv_input_stream;
class nv_output_stream;


// ----------------------------------------------------------------------------
// class nv_animation
//
struct DECLSPEC_NV_NVB nv_animation
{
    nv_animation();
    virtual ~nv_animation();

    unsigned int    num_keys;       // number of animation keyframes
    nv_scalar       freq;           // sampling frequency
                                    
    // animation tracks - null means not available
    quat          * rot;            // rotation
    vec3          * pos;            // position
    vec3          * scale;          // scale
};

        /// Write an animation to an nv_output_stream.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const nv_animation & rAnimation);

        /// Read an animation from an nv_input_stream.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_animation & rAnimation);


#endif // _nv_animation_h_
