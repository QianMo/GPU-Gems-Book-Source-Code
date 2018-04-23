// ================================================================
// $Id: Timers.h,v, 1.1.1.1 2004/10/12 12:49:13 matt Exp $
// ****************************************************************
// 
/** \file time.h

Various time measuring routines

@author Jiri Bittner
*/

#ifndef __TIMERS_H
#define __TIMERS_H

void initTiming();

void finishTiming();

long getTime();

double timeDiff(long t1,long t2);

char * timeString();


/** Example of usage:
see timers.h for diffirent possibilities of what is really measured..


InitTiming();

long t1, t2, t3;

t1 = GetTime();

Algortihm1();

t2 = GetTime();

Algorithm2();

t3 = GetTime();


cout<<"Algorithm 1"<<TimeDiff(t1, t2)<<"[ms]"<<endl;
cout<<"Algorithm 2"<<TimeDiff(t2, t3)<<"[ms]"<<endl;


FinishTiming();

*/


#endif
