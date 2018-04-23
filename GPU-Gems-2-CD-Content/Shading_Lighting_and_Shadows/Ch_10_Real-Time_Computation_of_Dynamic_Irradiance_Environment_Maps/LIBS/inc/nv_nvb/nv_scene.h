/*********************************************************************NVMH1****
File:
nv_scene.h

Copyright (C) 1999, 2002 NVIDIA Corporation
This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.

Comments:


******************************************************************************/

#ifndef _nv_scene_h_
#define _nv_scene_h_


//
// Forward declarations
//

class nv_input_stream;
class nv_output_stream;


// ----------------------------------------------------------------------------
// nv_scene class
//
struct DECLSPEC_NV_NVB nv_scene : public nv_refcount
{
    nv_scene();
    virtual ~nv_scene();

    nv_idx          find_node_idx(const nv_node * node);
    
    // visitor entry point
    virtual bool    accept(nv_visitor & visitor) const;

    char          * name;           // scene name

    // scene nodes - flattened
    unsigned int    num_nodes;      // number of nodes
    nv_node      ** nodes;          // array of nodes pointers

    // textures
    unsigned int    num_textures;   // number of textures
    nv_texture    * textures;       // array of nv_texture objects

    // materials
    unsigned int    num_materials;  // number of materials
    nv_material   * materials;      // array of materials

    // properties
    vec4            ambient;        // ambient color
    
    // scene bounding box information: models, lights, skeletons,...
    vec3            aabb_min;
    vec3            aabb_max;

    // models bounding box information: models only.
    vec3            models_aabb_min;
    vec3            models_aabb_max;
    // animation...
    unsigned int    num_keys;
};

        /// Write a scene to an nv_output_stream.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const nv_scene & rScene);

        /// Read a scene from an nv_input_stream.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_scene & rScene);



#endif // _nv_scene_h_
