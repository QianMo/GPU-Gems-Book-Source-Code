/*********************************************************************NVMH1****
File:
nv_node.h

Copyright (C) 1999, 2002 NVIDIA Corporation
This file is provided without support, instruction, or implied warranty of any
kind.  NVIDIA makes no guarantee of its fitness for a particular purpose and is
not liable under any circumstances for any damages or loss whatsoever arising
from the use or inability to use this file or items derived from it.

Comments:


******************************************************************************/

#ifndef _nv_node_h_
#define _nv_node_h_

#ifdef WIN32
#pragma warning (disable:4251) 
#endif


//
// Forward declarations
//

struct nv_scene;

struct nv_camera;
struct nv_model;
struct nv_light;

class nv_input_stream;
class nv_output_stream;


// ----------------------------------------------------------------------------
// class nv_node
//
struct DECLSPEC_NV_NVB nv_node : public nv_refcount
{
    //
    // Construction and destruction
    //

            /// Default constructor
    nv_node();

            /// Destructor
            virtual 
   ~nv_node();

    
    //
    // Public methods
    //

            /// Convert to camera
            nv_camera *
    asCamera();

            /// Convert to model
            nv_model *
    asModel();

            /// Convert to light
            nv_light *
    asLight();


    //
    // Public data
    //

    typedef enum _node_type
    {
        ANONYMOUS       = 0x00000000,
        GEOMETRY        = 0x00000001,
        LIGHT           = 0x00000002,
        CAMERA          = 0x00000003
    } node_type;

    static node_type type;          // to be set to ANONYMOUS
    virtual node_type   get_type() const; // return the node type

    virtual bool        accept(const nv_scene & scene, nv_visitor & visitor) const;
    
    char          * name;           // name of the node
    mat4            xform;          // model matrix transform, i.e.:
                                    // transforms a vertex into world space
    // hierarchy
    unsigned int    num_children;   // number of children
    nv_idx        * children;       // array of children - indices
                                    // into the scene node array
    nv_idx          parent;         // index to the parent - 
                                    // if parent == NV_BAD_IDX this is a root node
    nv_idx          target;         // index to the target - 
                                    // if target == NV_BAD_IDX there isn't any target
                                    // A target defines an "abstract relation" before
                                    // two nodes. For example, if "this" node is a camera
                                    // target defines the object the camera is looking at.
                                    // To look at more than one node, create a hierarchy
                                    // with the interested node as children, and target a
                                    // dummy parent node.

    // animation...
    nv_animation    anim;

    // abstract attibutes...
    nv_attribute    attr;
};

        /// Write a node to an nv_output_stream.
        DECLSPEC_NV_NVB 
        nv_output_stream &
operator << (nv_output_stream & rOutputStream, const nv_node & rNode);

        /// Read a node from an nv_input_stream.
        DECLSPEC_NV_NVB 
        nv_input_stream &
operator >> (nv_input_stream & rInputStream, nv_node & rNode);


#endif // _nv_node_h_
