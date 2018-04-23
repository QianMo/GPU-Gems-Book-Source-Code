/*********************************************************************NVMH1****
File:
nv_light.h

Copyright (C) 1999, 2002 NVIDIA Corporation
This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.

Comments:


******************************************************************************/

#ifndef _nv_light_h_
#define _nv_light_h_

struct DECLSPEC_NV_NVB nv_light : public nv_node
{
    nv_light();
    virtual ~nv_light();
    
    typedef enum _light_type
    {
        ANONYMOUS,
        POINT,
        DIRECTIONAL,
        SPOT
    } light_type;

    static node_type type;          // to be set to LIGHT
    virtual node_type get_type() const; // return the node type

    vec4            color;          // light color
    vec4            specular;       // light specular
    vec4            ambient;        // light ambient
    light_type      light;          
    nv_scalar       specular_exp;   // specular exponent

    // for point light, the node is giving the point location
    vec3            direction;      // direction in the local frame

    nv_scalar       range;          // light range

    // light attenuation:
    //           1
    // ----------------------
    // Kc + Kl * d + Kq * d^2
    nv_scalar       Kc;             // constant term
    nv_scalar       Kl;             // linear term
    nv_scalar       Kq;             // quadratic term

    // spot light properties...
    nv_scalar       falloff;        // spot falloff (d3d) or cutoff (ogl)
    nv_scalar       theta;          // inner cone angle - fully illuminated 
                                    // theta elt [0, phi]
    nv_scalar       phi;            // outer cone angle - fading out
};

        /// Write a light to an nv_output_stream.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const nv_light & rLight);

        /// Read a light from an nv_input_stream.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_light & rLight);

#endif // _nv_light_h_
