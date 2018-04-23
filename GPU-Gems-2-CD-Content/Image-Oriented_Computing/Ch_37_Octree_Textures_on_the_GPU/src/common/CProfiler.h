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
// -------------------------------------------------------------
//
// class CProfiler
//
//   Usefull to profile graphics app (D3D and GL)
//   Define D3D to enable DirectX support
//
// Author : Sylvain Lefebvre
// History: Created somewhere during 2002
//          2004-03-18 merge D3D and OpenGL versions
//
// OpenGL version uses libglfont (www.aracknea.net)
//
// -------------------------------------------------------------
#ifndef __CPROFILER__
#define __CPROFILER__
// -------------------------------------------------------------
#ifdef D3D
#include <Windows.h>
#include <d3d9.h>
#include <d3dx9.h>
#include <commctrl.h>
#include <d3dfont.h>
#else
#include <CFont.h>
#endif

#ifndef WIN32
typedef unsigned int DWORD;
#endif
// -------------------------------------------------------------
#include <stdio.h>
// -------------------------------------------------------------
#include <vector>
#include <string>
#include <iostream>
#include <map>
// -------------------------------------------------------------
#define PROFILER         (*CProfiler::getUniqueInstance())
#define DESTROY_PROFILER (*CProfiler::getUniqueInstance(true))
// -------------------------------------------------------------
#define PROFILER_HISTORY_MAX_SIZE 100
// -------------------------------------------------------------
class CProfilerVarData
{
protected:
  double      m_History[PROFILER_HISTORY_MAX_SIZE];
  double      m_dMax;
  double      m_dMin;
  int         m_iNext;
  std::string m_Name;
  DWORD       m_Color;
  bool        m_bInit;
  int         m_iUpdatePeriod;
  int         m_iLastUpdate;
  int         m_iGroup;
  FILE       *m_RecordFile;
  bool        m_bRecordInProgress;
  int         m_iRecordFrame;

  double&         history(int i);
  double          history(int i) const;

public:
  CProfilerVarData(std::string name,DWORD clr,int period);
  virtual ~CProfilerVarData() {}
	
  virtual double getValue()=0;
  virtual void   setValue(double)=0;

  virtual void   step();
  double         getPreviousValue(int age) const 
    {return (history(((m_iNext+PROFILER_HISTORY_MAX_SIZE-age))%PROFILER_HISTORY_MAX_SIZE));}
  std::string    getName()  const {return (m_Name);}
  const DWORD    getColor() const {return (m_Color);}
  double         getMin() const {return (m_dMin);}
  double         getMax() const {return (m_dMax);}
  bool           isValid() const {return (m_bInit);}
  void           normalize();
  virtual bool   isTime() const {return (false);}
  int            getGroup() const {return (m_iGroup);}
  void           setGroup(int g) {m_iGroup=g;}
  void           startRecord(const char *dir);
  void           stopRecord();
};
// -------------------------------------------------------------
template <typename T> class CProfilerVar : public CProfilerVarData
{
protected:
  T *m_Ptr;
public:
  CProfilerVar(std::string name,DWORD clr,T *ptr,int period)
    : CProfilerVarData(name,clr,period) {m_Ptr=ptr;}
  virtual ~CProfilerVar() {}
  virtual double getValue()         {return (double)(*m_Ptr);}
  virtual void   setValue(double v) {(*m_Ptr)=(T)(v);}
};
// -------------------------------------------------------------
class CProfilerTimer : public CProfilerVarData
{
protected:
  int m_dAccum;
  int m_dStart;
public:
  CProfilerTimer(std::string name,DWORD clr,int period)
    : CProfilerVarData(name,clr,period) {m_dAccum=0;m_dStart=0;}
  virtual ~CProfilerTimer() {}
  void   start();
  void   stop();
  virtual double getValue() {return (m_dAccum);}
  virtual void   setValue(double) {}
  virtual void   step() {CProfilerVarData::step(); m_dAccum=0;}
  virtual bool   isTime() const {return (true);}
};
// -------------------------------------------------------------
#define ADDVAR_DEF(T) void addVar(T *ptr,std::string name,DWORD clr,int group=0,int period=-1);
// -------------------------------------------------------------
class CProfiler
{
protected:
  std::vector<CProfilerVarData *>         m_Vars;
  std::map<int,std::pair<double,double> > m_GroupsMinMax;
  int                                     m_iTimerFreq;
  int                                     m_iDisplayGroup;

#ifdef D3D  
  LPDIRECT3DDEVICE9               m_pd3dDevice;
  CD3DFont                       *m_pFont;
  LPDIRECT3DVERTEXBUFFER9         m_pVB;
#else
  CFont                          *m_Font;
#endif

  CProfiler();
  ~CProfiler();

  void computeGroupsMinMax();

public:

#ifdef D3D  
  void init(LPDIRECT3DDEVICE9 d3d,CD3DFont *fnt);
#else
  void init(CFont *f) {m_Font=f;}
#endif
  void                       step();
  double                     getTime() const;
  double                     getRealTime() const {return (getTime()/m_iTimerFreq);}
  void                       draw();
  void                       normalize();
  int                        createTimer(std::string name,DWORD clr,int group=0);
  void                       startTimer(int tm);
  void                       stopTimer(int tm);
  std::pair<double,double>   getGroupMinMax(int g) {return (m_GroupsMinMax[g]);}
  void                       displayAllGroups() {m_iDisplayGroup=0;}
  void                       displayNextGroup();
  void                       startRecord(const char *dir=NULL);
  void                       stopRecord();

  ADDVAR_DEF(double);
  ADDVAR_DEF(float);
  ADDVAR_DEF(int);

  enum {blue  = 0x000000FF};
  enum {red   = 0x00FF0000};
  enum {green = 0x0000FF00};
  enum {yellow= 0x00FFFF00};
  enum {purple= 0x00FF00FF};

  static CProfiler *getUniqueInstance(bool destroy=false)
  {
    static CProfiler *profiler=NULL;
    if (destroy)
    {
      if (NULL != profiler)
        delete (profiler);
    }
    else
    {
      if (NULL == profiler)
        profiler=new CProfiler();
    }
    return (profiler);
  }
};
// -------------------------------------------------------------
#endif
// -------------------------------------------------------------
