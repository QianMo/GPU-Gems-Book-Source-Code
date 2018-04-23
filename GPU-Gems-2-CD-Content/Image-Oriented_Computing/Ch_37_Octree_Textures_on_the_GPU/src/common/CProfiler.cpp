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
// ---------------------------------------------------------------

#include "CProfiler.h"

// ---------------------------------------------------------------

#ifdef WIN32
# include <windows.h>
#else
# include <sys/time.h>
#endif

// ---------------------------------------------------------------

#ifndef min 
#define min(a,b) ((a)<(b)?(a):(b))
#endif
#ifndef max 
#define max(a,b) ((a)>(b)?(a):(b))
#endif

// ---------------------------------------------------------------

#include <cstdlib>
#include <cstdio>
#include <assert.h>

// ---------------------------------------------------------------

#ifdef D3D
struct PROFILER_VERTEX { D3DXVECTOR3 p; DWORD color; };
static DWORD PROFILER_FVF = D3DFVF_XYZ | D3DFVF_DIFFUSE;
#else
#include <GL/gl.h>
#endif

// ---------------------------------------------------------------

CProfilerVarData::CProfilerVarData(std::string name,DWORD clr,int period) 
{
  m_iNext=0;
  m_Name=name;
  for (int i=0;i<PROFILER_HISTORY_MAX_SIZE;i++)
    history(i)=0.0;
  m_Color=clr;
  m_bInit=false;
  m_iLastUpdate=0;
  m_iUpdatePeriod=period;
  m_iGroup=0;
  m_bRecordInProgress=false;
}

// ---------------------------------------------------------------

void               CProfilerVarData::step()
{
  int tm=(int)PROFILER.getTime();
  
  if (!m_bInit || m_iUpdatePeriod < 0 || (tm - m_iLastUpdate > m_iUpdatePeriod))
  {
    double v=getValue();
    if (v > m_dMax || !m_bInit)
      m_dMax=v;
    if (v < m_dMin || !m_bInit)
      m_dMin=v;
    history((m_iNext++) % PROFILER_HISTORY_MAX_SIZE)=v;
    m_iLastUpdate=tm;
    m_bInit=true;
  }

  // record
  if (m_bRecordInProgress)
  {
    fprintf(m_RecordFile,"%d %f\n",m_iRecordFrame,getValue());
    m_iRecordFrame++;
  }

}

// ---------------------------------------------------------------

void           CProfilerVarData::normalize()
{
  std::cerr << "=====> " << getName() << " <===== " << std::endl;
  std::cerr << "  min = " << m_dMin << ' ' << " max = " << m_dMax << std::endl;
  m_dMax=history(0);
  m_dMin=history(0);
  for (int i=1;i<PROFILER_HISTORY_MAX_SIZE;i++)
  {
    double v=history(i);
    if (v > m_dMax)
      m_dMax=v;
    if (v < m_dMin)
      m_dMin=v;    
  }
  std::cerr << "  over last " << PROFILER_HISTORY_MAX_SIZE << " frames: " << std::endl;
  std::cerr << "  min = " << m_dMin << ' ' << " max = " << m_dMax << std::endl;
  std::cerr << std::endl;
}

// ---------------------------------------------------------------

double&         CProfilerVarData::history(int i)
{
  assert(i >= 0 && i < PROFILER_HISTORY_MAX_SIZE);
  return (m_History[i]);
}

// ---------------------------------------------------------------

double         CProfilerVarData::history(int i) const
{
  assert(i >= 0 && i < PROFILER_HISTORY_MAX_SIZE);
  return (m_History[i]);
}

// ---------------------------------------------------------------

void           CProfilerVarData::startRecord(const char *dir)
{
  m_bRecordInProgress=true;
  m_iRecordFrame=0;
  std::string fname;
  if (dir)
    fname=std::string(dir)+m_Name+".plot";
  else
    fname=m_Name+".plot";
  m_RecordFile=fopen(fname.c_str(),"w");
  if (m_RecordFile == NULL)
    m_bRecordInProgress=false;
}

// ---------------------------------------------------------------

void           CProfilerVarData::stopRecord()
{
  if (m_bRecordInProgress)
  {
    m_bRecordInProgress=false;
    fclose(m_RecordFile);
  }
}

// ---------------------------------------------------------------
// ---------------------------------------------------------------
// ---------------------------------------------------------------

void   CProfilerTimer::start() 
{
  m_dStart=(int)PROFILER.getTime();
}

// ---------------------------------------------------------------

void   CProfilerTimer::stop()  
{
  m_dAccum+=((int)PROFILER.getTime()-m_dStart);
}

// ---------------------------------------------------------------
// ---------------------------------------------------------------
// ---------------------------------------------------------------

CProfiler::CProfiler()
{
  std::cerr << "Profiler initialized" << std::endl;
  m_iDisplayGroup=0;
#ifdef WIN32
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  m_iTimerFreq=(int)(freq.QuadPart/1000);
#else
  m_iTimerFreq=1;
#endif
#ifdef D3D
  m_pd3dDevice=NULL;
  m_pVB=NULL;
#endif
}

// ---------------------------------------------------------------

CProfiler::~CProfiler()
{
#ifdef D3D
  if (m_pVB != NULL)
    m_pVB->Release();
#endif
}

// ---------------------------------------------------------------

#ifdef D3D

void CProfiler::init(LPDIRECT3DDEVICE9 d3d,CD3DFont *fnt)
{
  HRESULT hr;

  m_pd3dDevice=d3d;
  m_pFont=fnt;
  if( FAILED( hr = m_pd3dDevice->CreateVertexBuffer( 
      PROFILER_HISTORY_MAX_SIZE * sizeof(PROFILER_VERTEX),
      D3DUSAGE_WRITEONLY | D3DUSAGE_DYNAMIC, 0,
      D3DPOOL_DEFAULT, &m_pVB, NULL ) ) )
  {
    MessageBox(NULL,"Cannot initialized Profiler !","Error",MB_OK | MB_ICONEXCLAMATION);
  }
}

#endif

// ---------------------------------------------------------------

void               CProfiler::step()
{
  // step vars
  for (unsigned int i=0;i<m_Vars.size();i++)
    m_Vars[i]->step();
  // update group's min/max
  computeGroupsMinMax();
}

// ---------------------------------------------------------------

void CProfiler::computeGroupsMinMax()
{
  double gmin=0,gmax=0;
  // global min/max
  for (unsigned int i=0;i<m_Vars.size();i++)
  {
    if (i==0)
    {
      gmin=m_Vars[i]->getMin();
      gmax=m_Vars[i]->getMax();
    }
    else
    {
      gmin=min(gmin,m_Vars[i]->getMin());
      gmax=max(gmax,m_Vars[i]->getMax());
    }
  }
  // update group's min/max values
  for (std::map<int,std::pair<double,double> >::iterator G=m_GroupsMinMax.begin();
    G != m_GroupsMinMax.end();G++)
  {
    (*G).second.first =gmax; // init min with global max
    (*G).second.second=gmin; // init max with global min
  }
  for (unsigned int i=0;i<m_Vars.size();i++)
  {
    m_GroupsMinMax[m_Vars[i]->getGroup()].first =min(m_GroupsMinMax[m_Vars[i]->getGroup()].first,m_Vars[i]->getMin());
    m_GroupsMinMax[m_Vars[i]->getGroup()].second=max(m_GroupsMinMax[m_Vars[i]->getGroup()].second,m_Vars[i]->getMax());
  }
}

// ---------------------------------------------------------------

double CProfiler::getTime() const
{
#ifdef WIN32
  LARGE_INTEGER tps;
  QueryPerformanceCounter(&tps);
  return ((double)tps.QuadPart);
  //return (timeGetTime());
#else
  struct timeval      now;
  unsigned int        ticks;

  static struct timeval start;
  static bool           init=false;

  if (!init)
  {
    gettimeofday(&start, NULL);
    init=true;
  }
  gettimeofday(&now, NULL);
  ticks=(now.tv_sec-start.tv_sec)*1000+(now.tv_usec-start.tv_usec)/1000;
  return (ticks);
#endif
}

// ---------------------------------------------------------------

#ifdef D3D
//        DirectX
void               CProfiler::draw()
{
  D3DVIEWPORT9 vp;
  m_pd3dDevice->GetViewport( &vp );

  m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP,   D3DTOP_SELECTARG2 );
  m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG1, D3DTA_TEXTURE );
  m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG2, D3DTA_DIFFUSE );

  m_pd3dDevice->SetFVF( PROFILER_FVF  );
  m_pd3dDevice->SetPixelShader( NULL );
  m_pd3dDevice->SetStreamSource( 0, m_pVB, 0, sizeof(PROFILER_VERTEX) );
  for (unsigned int i=0;i<m_Vars.size();i++)
  {
    if (m_Vars[i]->isValid() && (m_iDisplayGroup == 0 || m_iDisplayGroup == m_Vars[i]->getGroup()))
    {
      PROFILER_VERTEX* pVertices = NULL;

      m_pVB->Lock( 0, 0, (void**)&pVertices, D3DLOCK_DISCARD );
      for (unsigned int j=0;j<PROFILER_HISTORY_MAX_SIZE-1;j++)
      {
        pVertices[j].p.x=j/(float)(PROFILER_HISTORY_MAX_SIZE-1);
        if (m_Vars[i]->getGroup() != 0)
          pVertices[j].p.y=(float)(1.0-((m_Vars[i]->getPreviousValue(PROFILER_HISTORY_MAX_SIZE-1-j)-m_GroupsMinMax[m_Vars[i]->getGroup()].first)
				      /(m_GroupsMinMax[m_Vars[i]->getGroup()].second-m_GroupsMinMax[m_Vars[i]->getGroup()].first)));
        else
          pVertices[j].p.y=(float)(1.0-((m_Vars[i]->getPreviousValue(PROFILER_HISTORY_MAX_SIZE-1-j)-m_Vars[i]->getMin())
				      /(m_Vars[i]->getMax()-m_Vars[i]->getMin())));
        pVertices[j].p.z=0.0;
        pVertices[j].color=m_Vars[i]->getColor();
      }
      m_pVB->Unlock();
      m_pd3dDevice->DrawPrimitive( D3DPT_LINESTRIP, 0, j-1 );
    }
  }
  if (m_pFont != NULL)
  {
    for (unsigned int i=0;i<m_Vars.size();i++)
    {
      if (m_Vars[i]->isValid() && (m_iDisplayGroup == 0 || m_iDisplayGroup == m_Vars[i]->getGroup()))
      {
        static char str[64];
        if (m_Vars[i]->isTime())
          sprintf(str,"%s %.2f ms",m_Vars[i]->getName().c_str(),(m_Vars[i]->getPreviousValue(1)/((double)m_iTimerFreq)));
        else
          sprintf(str,"%s %.2f",m_Vars[i]->getName().c_str(),m_Vars[i]->getPreviousValue(1));
        SIZE size;
        m_pFont->GetTextExtent(str,&size);
        double y;
        if (m_Vars[i]->getGroup() != 0)
          y=1.0-((m_Vars[i]->getPreviousValue(1)-m_GroupsMinMax[m_Vars[i]->getGroup()].first)/(m_GroupsMinMax[m_Vars[i]->getGroup()].second-m_GroupsMinMax[m_Vars[i]->getGroup()].first));
        else
          y=1.0-((m_Vars[i]->getPreviousValue(1)-m_Vars[i]->getMin())/(m_Vars[i]->getMax()-m_Vars[i]->getMin()));
        m_pFont->DrawText((float)(vp.Width-size.cx),(float)min(y*vp.Height,vp.Height-size.cy),m_Vars[i]->getColor(),str);
      }
    }
  }
  m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP,   D3DTOP_MODULATE );
}
#else
//        OpenGL
void               CProfiler::draw()
{
  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_LIGHTING);
  glDisable(GL_TEXTURE_2D);
  for (unsigned int i=0;i<m_Vars.size();i++)
  {
    if (m_Vars[i]->isValid() && (m_iDisplayGroup == 0 || m_iDisplayGroup == m_Vars[i]->getGroup()))
    {
      DWORD clr=m_Vars[i]->getColor();
      DWORD r=(clr >> 16) & 255;
      DWORD g=(clr >>  8) & 255;
      DWORD b=(clr      ) & 255;
      glColor3f(r/255.0f,g/255.0f,b/255.0f);
      glBegin(GL_LINE_STRIP);
      for (unsigned int j=0;j<PROFILER_HISTORY_MAX_SIZE-1;j++)
      {
        if (m_Vars[i]->getGroup() != 0)
          glVertex2d(j/(double)(PROFILER_HISTORY_MAX_SIZE-1),
                    1.0-((m_Vars[i]->getPreviousValue(PROFILER_HISTORY_MAX_SIZE-1-j)-m_GroupsMinMax[m_Vars[i]->getGroup()].first)
				            /(m_GroupsMinMax[m_Vars[i]->getGroup()].second-m_GroupsMinMax[m_Vars[i]->getGroup()].first))          
          );
        else
          glVertex2d(j/(double)(PROFILER_HISTORY_MAX_SIZE-1),
	  	               1.0-((m_Vars[i]->getPreviousValue(PROFILER_HISTORY_MAX_SIZE-1-j)-m_Vars[i]->getMin())
		    	           /(m_Vars[i]->getMax()-m_Vars[i]->getMin())));
      }
      glEnd();
      
      if (m_Font != NULL)
      {
        static char str[64];
        if (m_Vars[i]->isTime())
          sprintf(str,"%s %.2f ms",m_Vars[i]->getName().c_str(),(m_Vars[i]->getPreviousValue(1)/((double)m_iTimerFreq)));
        else
          sprintf(str,"%s %.2f",m_Vars[i]->getName().c_str(),m_Vars[i]->getPreviousValue(1));
        double w,h;
        m_Font->printStringNeed(0.04,str,&w,&h);
        double y;
        if (m_Vars[i]->getGroup() != 0)
          y=1.0-((m_Vars[i]->getPreviousValue(1)-m_GroupsMinMax[m_Vars[i]->getGroup()].first)/(m_GroupsMinMax[m_Vars[i]->getGroup()].second-m_GroupsMinMax[m_Vars[i]->getGroup()].first));
        else
          y=1.0-((m_Vars[i]->getPreviousValue(1)-m_Vars[i]->getMin())/(m_Vars[i]->getMax()-m_Vars[i]->getMin()));
        m_Font->printString(1.0-w,min(y,1.0-0.04),0.04,str);
      }
    }
  }
  glPopAttrib();
}
#endif

// ---------------------------------------------------------------

void               CProfiler::displayNextGroup()
{
  static std::map<int,std::pair<double,double> >::iterator I=m_GroupsMinMax.begin();

  if (m_iDisplayGroup == 0)
    I=m_GroupsMinMax.begin();
  I++;
  if (I == m_GroupsMinMax.end())
    m_iDisplayGroup=0;
  else
    m_iDisplayGroup=(*I).first;
}

// ---------------------------------------------------------------

void               CProfiler::normalize()
{
  for (unsigned int i=0;i<m_Vars.size();i++)
    m_Vars[i]->normalize();
  // update group's min/max
  computeGroupsMinMax();
}

// ---------------------------------------------------------------

int                CProfiler::createTimer(std::string name,DWORD clr,int group)
{
  CProfilerTimer *tm=new CProfilerTimer(name,clr,-1);
  tm->setGroup(group);
  m_Vars.push_back(tm);
  return (m_Vars.size()-1);
}

// ---------------------------------------------------------------

void               CProfiler::startTimer(int tm)
{
  if (tm < 0)
    return;
  static_cast<CProfilerTimer *>(m_Vars[tm])->start();
}

// ---------------------------------------------------------------

void               CProfiler::stopTimer(int tm)
{
  if (tm < 0)
    return;
  static_cast<CProfilerTimer *>(m_Vars[tm])->stop();
}

// ---------------------------------------------------------------

void               CProfiler::startRecord(const char *dir)
{
  for (unsigned int i=0;i<m_Vars.size();i++)
    m_Vars[i]->startRecord(dir);
}

// ---------------------------------------------------------------

void               CProfiler::stopRecord()
{
  for (unsigned int i=0;i<m_Vars.size();i++)
    m_Vars[i]->stopRecord();
}

// ---------------------------------------------------------------

#define ADDVAR(T) void CProfiler::addVar(T *ptr,std::string name,DWORD clr,int group,int period) \
{ \
  CProfilerVar<T> *var=new CProfilerVar<T>(name,clr,ptr,period); \
  var->setGroup(group); \
	m_Vars.push_back(var); \
}

// ---------------------------------------------------------------

ADDVAR(double)
ADDVAR(float)
ADDVAR(int)

// ---------------------------------------------------------------
