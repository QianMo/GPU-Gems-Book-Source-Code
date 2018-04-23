/*********************************************************************NVMH1****
File:
nv_nvbfactory.h

Copyright (C) 1999, 2002 NVIDIA Corporation
This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.

Comments:


******************************************************************************/

#ifndef _nv_material_h_
#define _nv_material_h_

#ifdef WIN32
#pragma warning (disable:4251) 
#endif


struct DECLSPEC_NV_NVB nv_material
{
    nv_material();
    virtual ~nv_material();

    char          * name;           // name of the material
    unsigned int    id;             // material id

    // textures id to be looked up in the scene texture array
    unsigned int    num_textures;
    nv_idx        * textures;
    nv_idx        * tex_channel;    // idx of the texture coord set to be used 
                                    // for a given texture

    // material colors...
    vec4            diffuse;        // diffuse color
    vec4            specular;       // specular color
    nv_scalar       shininess;      // specular exponent
    vec4            ambient;        // ambient color
    vec4            emission;       // emitted light intensity of the material

    nv_scalar       transparent;    // level of transparency

    // fog properties...
    bool            fog;            // fog
    vec4            fog_color;      // fog color
    nv_attribute    attr;           // custom attributes
};

        /// Write a material to an nv_output_stream.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const nv_material & rMaterial);

        /// Read a material from an nv_input_stream.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_material & rMaterial);


#endif // _nv_material_h_
