/*********************************************************************NVMH1****
File:
nv_model.h

Copyright (C) 1999, 2002 NVIDIA Corporation
This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.

Comments:


******************************************************************************/

#ifndef _nv_model_h_ 
#define _nv_model_h_

struct DECLSPEC_NV_NVB nv_model : public nv_node
{
    nv_model();
    virtual ~nv_model();

    static node_type type;          // to be set to GEOMETRY
    virtual node_type get_type() const; // return the node type

    virtual bool    accept(const nv_scene & scene, nv_visitor & visitor) const;

    // meshes
    unsigned int    num_meshes;     // number of meshes
    nv_mesh       * meshes;         // array of meshes
    
    // bounding box information...
    vec3            aabb_min;
    vec3            aabb_max;
};

        /// Write a model to an nv_output_stream.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const nv_model & rModel);

        /// Read a model from an nv_input_stream.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_model & rModel);


#endif // _nv_model_h_
