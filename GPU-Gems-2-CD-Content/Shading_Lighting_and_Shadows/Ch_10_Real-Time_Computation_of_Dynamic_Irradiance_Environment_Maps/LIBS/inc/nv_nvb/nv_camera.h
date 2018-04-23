/*********************************************************************NVMH1****
File:
nv_camera.h

Copyright (C) 1999, 2002 NVIDIA Corporation
This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.

Comments:


******************************************************************************/

#ifndef _nv_camera_h_
#define _nv_camera_h_

struct DECLSPEC_NV_NVB nv_camera : public nv_node
{
    nv_camera();
    virtual ~nv_camera();

    typedef enum _camera_type
    {
        ANONYMOUS,
        POINT,
        DIRECTIONAL,
        SPOT
    } camera_type;

    static node_type type;          // to be set to CAMERA
    virtual node_type get_type() const; // return the node type

    nv_scalar       fov;            // field of view
    nv_scalar       focal_length;   // focal length
};

        /// Write a camera to an nv_output_stream.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const nv_camera & rCamera);

        /// Read a camera from an nv_input_stream.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_camera & rCamera);


#endif // _nv_camera_h_
