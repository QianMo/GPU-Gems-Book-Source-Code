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

#ifndef _nv_texcoord_set_h_
#define _nv_texcoord_set_h_

struct DECLSPEC_NV_NVB nv_texcoord_set
{
    nv_texcoord_set();
    virtual ~nv_texcoord_set();
    
    unsigned int    num_coords;
    unsigned int    dim;            // dimension of the set: 1D, 2D, 3D,...
    nv_scalar     * texcoords;      // array of texture coords of dimension dim
                                    // example: for a 2D texture coordinate set:
                                    // dim = 2 and 
                                    // [s,t,s,t,s,t,..., nv_mesh->num_vertices * 2]

    // texture basis
    // note : use the normals from the geometry to complete the basis...
    vec3          * binormals;
    vec3          * tangents;
};

        /// Write a texcoord_set to an nv_output_stream.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const nv_texcoord_set & rTexCoordSet);

        /// Read a texcoord_set from an nv_input_stream.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_texcoord_set & rTexCoordSet);



#endif // _nv_texcoord_set_h_
