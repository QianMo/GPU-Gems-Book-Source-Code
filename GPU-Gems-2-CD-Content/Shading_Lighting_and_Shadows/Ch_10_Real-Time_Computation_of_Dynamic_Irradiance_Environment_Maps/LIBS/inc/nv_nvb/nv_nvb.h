/*********************************************************************NVMH4****
Path:  SDK\LIBS\inc\nv_nvb
File:  nv_nvb.h

Copyright NVIDIA Corporation 2002
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.



Comments:


******************************************************************************/

#ifndef _nv_nvb_h_
#define _nv_nvb_h_

#ifdef WIN32
#pragma warning (disable:4786) 
#endif

#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <queue>
#include <algorithm>

#ifndef _nv_mathdecl_h_
#include <nv_math/nv_math.h>
#endif // _nv_mathdecl_h_

#ifndef _nv_nvbdecl_h_
#include "nv_nvbdecl.h"
#endif // _nv_nvbdecl_h_


#include "nv_streams.h"
#include "nv_file.h"


#ifndef _nv_core_h_
#include <nv_nvb/nv_core.h>
#endif // _nv_core_h_

#ifdef WIN32
    #ifndef NV_NVB_PROJECT // defined if we are building nv_nvb.lib
        #ifndef NV_NVB_EXPORTS
            #if _MSC_VER >= 1300
                #ifdef _DLL
                    #pragma message("Note: including lib: nv_nvb.lib\n")
                    #pragma comment(lib,"nv_nvb.lib")
                #else
                    #error "Your project doesn't use the Multithreaded DLL Runtime"
                #endif
            #endif
        #endif // NV_NVB_EXPORTS
    #endif // NV_NVB_PROJECT
#endif // WIN32

typedef unsigned int    nv_idx;             // index type

#define NV_BAD_IDX      0xFFFFFFFF


#ifndef _nv_refcount_h_
#include "nv_refcount.h"
#endif // _nv_refcount_h_

#ifndef _nv_animation_h_
#include "nv_animation.h"
#endif // _nv_animation_h_

#ifndef _nv_visitor_h_
#include "nv_visitor.h"
#endif // _nv_visitor_h_

#ifndef _nv_node_h_
#include "nv_node.h"
#endif // _nv_node_h_

#ifndef _nv_texture_h_
#include "nv_texture.h"
#endif // _nv_texture_h_

#ifndef _nv_texcoord_set_h_
#include "nv_texcoord_set.h"
#endif // _nv_texcoord_set_h_

#ifndef _nv_material_h_
#include "nv_material.h"
#endif // _nv_material_h_

#ifndef _nv_camera_h_
#include "nv_camera.h"
#endif // _nv_camera_h_

#ifndef _nv_light_h_
#include "nv_light.h"
#endif // _nv_light_h_

#ifndef _nv_mesh_h_
#include "nv_mesh.h"
#endif // _nv_mesh_h_

#ifndef _nv_model_h_
#include "nv_model.h"
#endif // _nv_model_h_

#ifndef _nv_scene_h_
#include "nv_scene.h"
#endif // _nv_scene_h_

#ifndef _nv_drawmodel_visitor_h_
#include "nv_drawmodel_visitor.h"
#endif // _nv_drawmodel_visitor_h_

#ifndef _nv_factory_h_
#include "nv_factory.h"
#endif // _nv_factory_h_

#include "nv_streams.h"
#include "nv_file.h"

#define     NVB_LHS     0x00000001
#define     NVB_RHS     0x00000002

bool    DECLSPEC_NV_NVB NVBLoad(const char * file, nv_scene * scene, long options = NVB_RHS);
void    DECLSPEC_NV_NVB NVBSetLogCallback(void* cbfn, unsigned long userparam=0);


        // changeHandedness
        //
        // Description:
        //      Convert scene data's handedness.
        //          Depending on what graphics library is used it
        //      is more convenient to have all sence data specified with
        //      respect to a righthanded coordinate system, as opposed to
        //      left-handed coordinate system.
        //          This is a helper method that converts the handedness of
        //      a scene in place.
        // 
        // Parameters:
        //      pScene - pointer to the nv_scene to be changed.
        //
        // Returns:
        //      true  - on success,
        //      flase - otherwise.
        //
        bool
changeHandedness(nv_scene * pScene);

        // changeHandedness
        //
        // Description:
        //      Convert an nv_node's handedness.
        //          Nodes contain a transform matrix that needs to
        //      be changed depending on the handedness of the coordinate
        //      system.
        //
        // Parameters:
        //      pNode - pointer to the nv_node to be changed.
        //
        // Returns:
        //      true  - on success,
        //      false - otherwise.
        //
        bool
changeHandedness(nv_node * pNode);

        // changeHandedness
        //
        // Description:
        //      Convert an nv_model's handedness.
        //          All positions and normals switch handedness.
        //
        // Parameters:
        //      pModel - pointer to the nv_model node to be changed.
        //
        // Returns:
        //      true  - on success,
        //      false - otherwise.
        //
        bool
changeHandedness(nv_model * pModel);

        // changeHandedness
        //
        // Description:
        //      Convert an nv_mesh's handedness.
        //          All positions and normals switch handedness.
        //
        // Parameters:
        //      pMesh - pointer to the nv_mdesh to be changed.
        //
        // Returns:
        //      true  - on success,
        //      false - otherwise.
        //
        bool
changeHandedness(nv_mesh * pMesh);

        // changeHandedness
        //
        // Description:
        //      Convert an nv_light's handedness.
        //          Directional lights store the light direction
        //      as a vector.
        //
        // Parameters:
        //      pLight - pointer to the nv_light to be changed.
        //
        // Returns:
        //      true  - on success,
        //      flase - otherwise.
        //
        bool
changeHandedness(nv_light * pLight);

        // changeHandedness
        //
        // Description:
        //      Convert an nv_camera's handedness.
        //
        // Parameters:
        //      pCamera - pointer to the nv_camera to be changed.
        //
        // Returns:
        //      true  - on success,
        //      flase - otherwise.
        //
        bool
changeHandedness(nv_camera * pCamera);

        // changeHandedness
        //
        // Description:
        //      Convert an animation's handedness.
        //
        // Parameters:
        //      pAnimation - pointer to the nv_animation to be changed.
        //
        // Returns:
        //      true  - on success,
        //      flase - otherwise.
        //
        bool
changeHandedness(nv_animation * pAnimation);

        // changeHandedness
        //
        // Description:
        //      Convert an mat4's handedness.
        //
        // Parameters:
        //      pMatrix - pointer to the mat4 to be changed.
        //
        // Returns:
        //      true  - on success,
        //      false - otherwise.
        //
        bool
changeHandedness(mat4 * pMatrix);

        // changeHandedness
        //
        // Description:
        //      Convert a quat's handedness.
        //
        // Parameters:
        //      pQuaternion - pointer to the quat to be changed.
        //
        // Returns:
        //      true  - on success,
        //      flase - otherwise.
        //
        bool
changeHandedness(quat * pQuaterion);


#endif //_nv_nvb_h_
