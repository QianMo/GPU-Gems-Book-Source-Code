/*********************************************************************NVMH1****
File:
nv_texture.h

Copyright (C) 1999, 2002 NVIDIA Corporation
This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.

Comments:


******************************************************************************/

#ifndef _nv_texture_h_
#define _nv_texture_h_

struct DECLSPEC_NV_NVB nv_texture
{
    nv_texture();
    nv_texture(const nv_texture & tex);
    virtual ~nv_texture();

    const nv_texture & operator= (const nv_texture & tex);
    void copy_from(const nv_texture & tex);

    // defines the different types of textures
    typedef enum _tex_type
    {
        CUSTOM              = 0x00000000,
        AMBIENT             = 0x00000001,
        DIFFUSE             = 0x00000002,
        SPECULAR            = 0x00000003, // specular color map
        SPECULAR_POWER      = 0x00000004, // specular highlight map
        GLOSS               = 0x00000005,
        SELF_ILLUMATION     = 0x00000006,
        BUMP                = 0x00000007,
        NORMAL              = 0x00000008,
        OPACITY             = 0x00000009,
        REFLECTION          = 0x0000000A,
        REFRACTION          = 0x0000000B,
        CUBE_MAP            = 0x0000000C
    } tex_type;

    tex_type        type;
    char          * name;           // texture file name
    mat4            tex_mat;        // texture matrix
};

        /// Write a scene to an nv_output_stream.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const nv_texture & rTexture);

        /// Read a scene from an nv_input_stream.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_texture & rTexture);


#endif // _nv_texture_h_
