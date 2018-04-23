/*********************************************************************NVMH1****
File:
nv_meshh

Copyright (C) 1999, 2002 NVIDIA Corporation
This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.

Comments:


******************************************************************************/

#ifndef _nv_mesh_h_
#define _nv_mesh_h_

#ifdef WIN32
#pragma warning (disable:4251) 
#endif

struct DECLSPEC_NV_NVB nv_mesh
{
    nv_mesh();
    virtual ~nv_mesh();

    nv_idx          material_id;    // material id - reference in the 
                                    // scene material array
    // geometry...
    unsigned int    num_vertices;   // number of vertices
    vec3          * vertices;       // array of vertices
    vec3          * normals;        // array of normals
    vec4          * colors;         // array of colors (vertex colors)

    // skinning...
    bool            skin;           // flag to tell if the vertices are to be skinned
    vec4          * weights;        // vertex weights
    nv_idx        * bone_idxs;      // 4 bones per vertex.

    // array of texcoord sets...
    unsigned int    num_texcoord_sets; // number of texture coordinate sets
    nv_texcoord_set * texcoord_sets; // array of texcoord_sets - they all contain 
                                    // num_vertices texture coordinates per set
    // topology...
    unsigned int    num_faces;      // numbers of triangle face
    nv_idx        * faces_idx;      // array of vertex indices - faces

    // custom vertex attributes
    nv_attribute    mesh_attr;      // attributes
    
    // bounding box information...
    vec3            aabb_min;
    vec3            aabb_max;


    //
    // Public methods
    //

            // calculateTangentSpaces
            //
            // Description:
            //      Calculates tangent-space information for a UV set.
            //          The tangen-space information consists of the
            //      per-vertex tangent and binormal vectors stored with
            //      each UV set.
            //
            // Parameters:
            //      iUVSet - UV set index.
            //
            // Returns:
            //      true - on success,
            //      flase - otherwise.
            //
            bool
    calculateTangentSpaces(unsigned int iUVSet);

};

        /// Write a mesh to an nv_output_stream.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const nv_mesh & rMesh);

        /// Read a mesh from an nv_input_stream.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_mesh & rMesh);



#endif // _nv_mesh_h_
