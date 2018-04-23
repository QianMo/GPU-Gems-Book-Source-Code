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
#ifndef __BOX__
#define __BOX__

#include "CPolygon.h"
#include "MgcBox3.h"

class COBox
{
protected:

  Mgc::Box3  m_MgcBox;

  CVertex    m_P,m_U,m_V,m_W;
  CVertex    m_InvU,m_InvV,m_InvW;

  CVertex toWorld(const CVertex& p) const
    {
      return (m_U*p.x() + m_V*p.y() + m_W*p.z() + m_P);
    }
  CVertex toLocal(const CVertex& p) const
    {
      CVertex tmp=p-m_P;
      return (CVertex(tmp.dot(m_InvU),tmp.dot(m_InvV),tmp.dot(m_InvW)));
    }

  void computeMgcBox();
  void computeMgcBox(const CVertex& u,
		     const CVertex& v,
		     const CVertex& w,
		     double lu,
		     double lv,
		     double lw);
  

  void init(const CVertex& p,
	    const CVertex& u,
	    const CVertex& v,
	    const CVertex& w,
	    double lu,
	    double lv,
	    double lw);
  void init(const CVertex& p,
	    const CVertex& u,
	    const CVertex& v,
	    const CVertex& w);

  void insert(const CVertex& p);

public:

  COBox();
  COBox(const COBox& b);
  COBox(const CVertex& p,
	const CVertex& u,
	const CVertex& v,
	const CVertex& w) {init(p,u,v,w);}
  
  COBox sub(const CVertex& local_pos,const CVertex& local_size) const;

  void cut(const CPolygon& p,CPolygon& _r) const;
  void draw_box_line() const;
  void draw_box_fill() const;

  CVertex center() const {return (toWorld(CVertex(0.5,0.5,0.5)));}

  CVertex p000() const {return (toWorld(CVertex(0,0,0)));}
  CVertex p001() const {return (toWorld(CVertex(0,0,1)));}
  CVertex p010() const {return (toWorld(CVertex(0,1,0)));}
  CVertex p011() const {return (toWorld(CVertex(0,1,1)));}
  CVertex p100() const {return (toWorld(CVertex(1,0,0)));}
  CVertex p101() const {return (toWorld(CVertex(1,0,1)));}
  CVertex p110() const {return (toWorld(CVertex(1,1,0)));}
  CVertex p111() const {return (toWorld(CVertex(1,1,1)));}

  CPlane pl0() const {return (CPlane(p000(),p110(),p010()));}
  CPlane pl1() const {return (CPlane(p001(),p111(),p101()));}
  CPlane pl2() const {return (CPlane(p010(),p001(),p000()));}
  CPlane pl3() const {return (CPlane(p100(),p111(),p110()));}
  CPlane pl4() const {return (CPlane(p000(),p101(),p100()));}
  CPlane pl5() const {return (CPlane(p010(),p111(),p011()));}

  CPolygon poly0() const {return CPolygon(p000(),p010(),p110(),p100());}
  CPolygon poly1() const {return CPolygon(p001(),p101(),p111(),p011());}
  CPolygon poly2() const {return CPolygon(p010(),p000(),p001(),p011());}
  CPolygon poly3() const {return CPolygon(p100(),p110(),p111(),p101());}
  CPolygon poly4() const {return CPolygon(p000(),p100(),p101(),p001());}
  CPolygon poly5() const {return CPolygon(p010(),p011(),p111(),p110());}

  friend inline std::ostream& operator<<(std::ostream& s,const COBox& b);
  friend inline std::istream& operator>>(std::istream& s,COBox& b);
  friend bool collide(const COBox *,const COBox *);
};

inline std::ostream& operator<<(std::ostream& s,const COBox& b)
{
  return (s << ' ' << b.m_P
	  << ' ' << b.m_U
	  << ' ' << b.m_V
	  << ' ' << b.m_W);
}

inline std::istream& operator>>(std::istream& s,COBox& b)
{
  s >> b.m_P;
  s >> b.m_U;
  s >> b.m_V;
  s >> b.m_W;
  b.init(b.m_P,b.m_U,b.m_V,b.m_W);
  return (s);
}

bool collide(const COBox *,const COBox *);

#endif
