// profiler.hpp
#ifndef __PROFILER_HPP__
#define __PROFILER_HPP__

#include <iostream>

// Uncomment this line to enable profiler information for the brook runtime...
//#define BROOK_PROFILER_ENABLED
#define BROOK_PROFILER_OUTPUT_PATH "./brookProfilerDump.txt"

#ifdef BROOK_PROFILER_ENABLED
#define	BROOK_PROFILE( __name ) \
  static ::brook::internal::ProfilerNode __profiler_node( __name ); \
  ::brook::internal::ProfilerSample __profiler_sample( __profiler_node );
#else
#define	BROOK_PROFILE( __name )	
#endif

#ifdef WIN32
typedef __int64 int64;
#else
typedef long long int64;
#endif

namespace brook {
namespace internal {

  class ProfilerNode
  {
  public:
    ProfilerNode( const char* inName );
    ~ProfilerNode();

    void enter();
    void exit();

    void pause();
    void resume();

    void dump( std::ostream& output );

  private:
    friend class Profiler;
    const char* name;
    ProfilerNode* nextNode;

    int64 totalCalls;

    int64 totalTime;
    int64 withoutChildrenTime;

    int64 totalTimeStart;
    int64 withoutChildrenTimeStart;

    int64 recursionDepth;
  };

  class Profiler
  {
  public:
    static Profiler& getInstance();
    void addNode( ProfilerNode* inNode );

  private:
    Profiler();


    void dump();

    ProfilerNode* firstNode;
#ifdef _MSC_VER
#if _MSC_VER <= 1200
  public: //bug with destructor protection. Fails to compile with private des
#endif
#endif
    ~Profiler();
  };

  class ProfilerSample
  {
  public:
    ProfilerSample( ProfilerNode& inNode )
      : node(inNode), saved(sCurrentNode)
    {
      if(saved)
        saved->pause();
      sCurrentNode = &node;
      node.enter();
    }
    ~ProfilerSample()
    {
      node.exit();
      sCurrentNode = saved;
      if(saved)
        saved->resume();
    }

  private:
    ProfilerNode& node;
    ProfilerNode* saved;
    static ProfilerNode* sCurrentNode;
  };


}}

#endif
