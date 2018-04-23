// profiler.cpp
#include "profiler.hpp"

#ifndef _WIN32

#include <unistd.h>
#include <sys/time.h>
#include <string.h>

static int64 getFrequency() {
  return (int64)1000000;
}

static int64 getTime()
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  int64 temp = tv.tv_usec;
  temp+= tv.tv_sec*1000000;
  return temp;
}

#else

#include <windows.h>

static int64 getFrequency()
{
  LARGE_INTEGER frequency;
  if( !QueryPerformanceFrequency(&frequency) )
  {
    std::cerr << "Unable to read the performance counter frequency!\n";
    return 0;
  }

  return (int64)(frequency.QuadPart);
}

static int64 getTime()
{
  LARGE_INTEGER counter;
  if( !QueryPerformanceCounter(&counter) )
  {
    std::cerr << "Unable to read the performance counter!\n";
    return 0;
  }

  return (int64)(counter.QuadPart);
}

#endif

#include <fstream>


namespace brook {
namespace internal {

  ProfilerNode* ProfilerSample::sCurrentNode = NULL;
  
  ProfilerNode::ProfilerNode( const char* inName )
    : name(inName),
    totalCalls(0),
    totalTime(0),
    withoutChildrenTime(0),
    totalTimeStart(0),
    withoutChildrenTimeStart(0),
    recursionDepth(0)
  {
    Profiler::getInstance().addNode( this );
  }

  ProfilerNode::~ProfilerNode() {
  }

  void ProfilerNode::enter()
  {
    totalCalls++;
    if( recursionDepth++ == 0 )
    {
      int64 time = getTime();
      totalTimeStart = time;
      withoutChildrenTimeStart = time;
    }
  }

  void ProfilerNode::exit()
  {
    if( recursionDepth-- == 1 )
    {
      int64 time = getTime();
      totalTime += time - totalTimeStart;
      withoutChildrenTime += time - withoutChildrenTimeStart;
    }
  }

  void ProfilerNode::pause()
  {
      int64 time = getTime();
      withoutChildrenTime += time - withoutChildrenTimeStart;
  }

  void ProfilerNode::resume()
  {
      int64 time = getTime();
      withoutChildrenTimeStart = time;
  }

  void ProfilerNode::dump( std::ostream& output )
  {
    // convert to microseconds
    double total = ((double)totalTime * 1000000.) / getFrequency();
    double withoutChildren = 
       ((double)withoutChildrenTime * 1000000.)
       / getFrequency();

    output << name << "\t" << (double)totalCalls << "\t" << total << "\t" << withoutChildren << std::endl;
  }

  Profiler::Profiler()
    : firstNode(0)
  {
  }

  Profiler::~Profiler()
  {
    dump();
  }

  Profiler& Profiler::getInstance()
  {
    static Profiler sResult;
    return sResult;
  }

  void Profiler::addNode( ProfilerNode* inNode )
  {
    inNode->nextNode = firstNode;
    firstNode = inNode;
  }

  void Profiler::dump()
  {
    std::ofstream output( BROOK_PROFILER_OUTPUT_PATH );
    ProfilerNode* node = firstNode;
    while( node )
    {
      node->dump( output );
      node = node->nextNode;
    }
  }

}}
