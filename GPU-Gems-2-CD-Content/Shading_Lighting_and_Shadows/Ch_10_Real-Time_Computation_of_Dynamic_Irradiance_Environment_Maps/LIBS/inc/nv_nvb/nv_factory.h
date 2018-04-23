/*********************************************************************NVMH4****
Path:  SDK\LIBS\inc\nv_nvb
File:  nv_factory.h

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

#ifndef _nv_factory_h_
#define _nv_factory_h_

class DECLSPEC_NV_NVB nv_factory
{
public:
    static nv_factory *     get_factory();
    static bool             shutdown();

    nv_scene *              new_scene();
    nv_node *               new_node();
    nv_model *              new_model();
    nv_light *              new_light();
    nv_camera *             new_camera();
    nv_drawmodel_visitor *  new_drawmodel_visitor();

    
protected:
    nv_factory();

    static nv_factory *     _the_factory;
};

#endif // _nv_factory_h_
