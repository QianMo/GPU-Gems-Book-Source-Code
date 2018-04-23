/* ----------------------------------------------------------

Octree Textures on the GPU - source code - GPU Gems 2 release
                                                   2004-11-21

Updates on http://www.aracknea.net/octreetex
--
(c) 2004 Sylvain Lefebvre - all rights reserved
--
The source code is provided 'as it is', without any warranties. 
Use at your own risk. The use of any part of the source code in a
commercial or non commercial product without explicit authorisation
from the author is forbidden. Use for research and educational
purposes is allowed and encouraged, provided that a short notice
acknowledges the author's work.
---------------------------------------------------------- */
#ifdef WIN32
#  include <windows.h>
#endif

#define REVERT_CUT

#include "CBox.h"
#include "common.h"

#include "MgcIntr3DBoxBox.h"

COBox::COBox() 
{
  init(CVertex(0,0,0),
       CVertex(1,0,0),
       CVertex(0,1,0), 
       CVertex(0,0,1));
}

void COBox::init(const CVertex& p,
		 const CVertex& u,
		 const CVertex& v,
		 const CVertex& w,
		 double lu,
		 double lv,
		 double lw)
{
  m_P=p;
  m_U=u*(vertex_real)lu;
  m_V=v*(vertex_real)lv;
  m_W=w*(vertex_real)lw;

  m_InvU=CVertex(
    u.x()/(vertex_real)lu,
    u.y()/(vertex_real)lu,
    u.z()/(vertex_real)lu);
  m_InvV=CVertex(
    v.x()/(vertex_real)lv,
    v.y()/(vertex_real)lv,
    v.z()/(vertex_real)lv);
  m_InvW=CVertex(
    w.x()/(vertex_real)lw,
    w.y()/(vertex_real)lw,
    w.z())/(vertex_real)lw;
  computeMgcBox(u,v,w,lu,lv,lw);
}

void COBox::init(const CVertex& p,
		 const CVertex& u,
		 const CVertex& v,
		 const CVertex& w)
{
  CVertex nu=u;
  CVertex nv=v;
  CVertex nw=w;
  double  lu=nu.normalize();
  double  lv=nv.normalize();
  double  lw=nw.normalize();
  init(p,nu,nv,nw,lu,lv,lw);
}

COBox::COBox(const COBox& b)
{
  m_P=b.m_P;
  m_U=b.m_U;
  m_V=b.m_V;
  m_W=b.m_W;
  m_InvU=b.m_InvU;
  m_InvV=b.m_InvV;
  m_InvW=b.m_InvW;
  m_MgcBox=b.m_MgcBox;
}

COBox COBox::sub(const CVertex& local_pos,
		 const CVertex& local_size) const
{
  COBox b(m_P+m_U*local_pos.x()
	  +m_V*local_pos.y()
	  +m_W*local_pos.z(),
	  m_U*local_size.x(),
	  m_V*local_size.y(),
	  m_W*local_size.z());
  return (b);
}

void COBox::cut(const CPolygon& p,CPolygon& _r) const
{
  CPolygon front,back0,back1;
#ifdef REVERT_CUT
  p.cut(    pl0(),back0,front);
  back0.cut(pl1(),back1,front);
  back1.cut(pl2(),back0,front);
  back0.cut(pl3(),back1,front);
  back1.cut(pl4(),back0,front);
  back0.cut(pl5(),_r,front);
#else
  p.cut(    pl0(),front,back0);
  back0.cut(pl1(),front,back1);
  back1.cut(pl2(),front,back0);
  back0.cut(pl3(),front,back1);
  back1.cut(pl4(),front,back0);
  back0.cut(pl5(),front,_r);
#endif
}

void COBox::draw_box_line() const
{
  glBegin(GL_LINES);

  p000().gl();
  p001().gl();

  p000().gl();
  p010().gl();

  p000().gl();
  p100().gl();

  p001().gl();
  p011().gl();
//
  p001().gl();
  p101().gl();

  p010().gl();
  p011().gl();

  p010().gl();
  p110().gl();

  p100().gl();
  p110().gl();
//
  p100().gl();
  p101().gl();

  p111().gl();
  p101().gl();

  p111().gl();
  p110().gl();

  p111().gl();
  p011().gl();

  glEnd();
}

void COBox::draw_box_fill() const
{
  poly0().gl();
  poly1().gl();
  poly2().gl();
  poly3().gl();
  poly4().gl();
  poly5().gl();
}

void COBox::computeMgcBox(const CVertex& u,
			  const CVertex& v,
			  const CVertex& w,
			  double lu,
			  double lv,
			  double lw)
{
  CVertex c=center();
  m_MgcBox.Center().x=c.x();
  m_MgcBox.Center().y=c.y();
  m_MgcBox.Center().z=c.z();

  m_MgcBox.Axis(0).x=u.x();
  m_MgcBox.Axis(0).y=u.y();
  m_MgcBox.Axis(0).z=u.z();
  m_MgcBox.Extents()[0]=(Mgc::Real)(lu*0.5);

  m_MgcBox.Axis(1).x=v.x();
  m_MgcBox.Axis(1).y=v.y();
  m_MgcBox.Axis(1).z=v.z();
  m_MgcBox.Extents()[1]=(Mgc::Real)(lv*0.5);

  m_MgcBox.Axis(2).x=w.x();
  m_MgcBox.Axis(2).y=w.y();
  m_MgcBox.Axis(2).z=w.z();
  m_MgcBox.Extents()[2]=(Mgc::Real)(lw*0.5);
}

void COBox::computeMgcBox()
{
  CVertex nu=m_U;
  CVertex nv=m_V;
  CVertex nw=m_W;
  double  lu=nu.normalize();
  double  lv=nv.normalize();
  double  lw=nw.normalize();
  computeMgcBox(nu,nv,nw,lu,lv,lw);
}

bool collide(const COBox *b0,const COBox *b1)
{
  /*
    if (   b0->m_Max.x() < b1->m_Min.x()
    || b0->m_Max.y() < b1->m_Min.y()
    || b0->m_Max.z() < b1->m_Min.z()
    || b1->m_Max.x() < b0->m_Min.x()
    || b1->m_Max.y() < b0->m_Min.y()
    || b1->m_Max.z() < b0->m_Min.z())
    return (false);
  */
  return (Mgc::TestIntersection(b0->m_MgcBox,b1->m_MgcBox));
}
