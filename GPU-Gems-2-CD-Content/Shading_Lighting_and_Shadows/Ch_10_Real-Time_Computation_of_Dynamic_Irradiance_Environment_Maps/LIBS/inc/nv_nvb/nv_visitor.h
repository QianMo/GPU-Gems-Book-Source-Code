/*********************************************************************NVMH4****
Path:  SDK\LIBS\inc\nv_nvb
File:  nv_visitor.h

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


Comments: Implement base class for visitor pattern.


******************************************************************************/

#ifndef _nv_visitor_h_
#define _nv_visitor_h_

// forward declarations
struct nv_model;
struct nv_node;

// nv_visitor class
class DECLSPEC_NV_NVB nv_visitor : public nv_refcount
{
public:
    virtual ~nv_visitor();
    virtual void visit_node(const nv_node * node);
    virtual void visit_model(const nv_model * model);
protected:
    nv_visitor();
};

#endif // _nv_visitor_h_
