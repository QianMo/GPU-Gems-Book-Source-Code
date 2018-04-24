//
// seqencegen.h
// Last Updated:		05.01.07
// 
// Mark Colbert & Jaroslav Krivanek
// colbert@cs.ucf.edu
//
// Copyright (c) 2007.
//
// The following code is freely distributed "as is" and comes with 
// no guarantees or required support by the authors.  Any use of 
// the code for commercial purposes requires explicit written consent 
// by the authors.
//

#include <Cg/cg.h>
#include <Cg/cgGL.h>

/// CG Global Parameters
extern CGparameter lScalesParam, lSmplsParam, wScalesParam, wSmplsParam;

extern float *randnum, *randcos, *randsin, *randlog;

enum SequenceType { HALTON=0, FOLDED_HALTON=1, HAMMERSLEY=2, FOLDED_HAMMERSLEY=3, 
					POSSION_DISK=4, BEST_CANDIDATE=5, PENROSE=6 };
extern SequenceType currSequenceType;

extern int samples;	///< Number of currently generated samples

/// Generates a quasi-random number sequence using currSequenceType
extern void genSequence(int n);

/// Generates Lafortune sample directions using the quasi-random numbers generated
extern void genLafortuneSamples(int smpls, float cxy, float cz, float n);

/// Generates Ward sample directions using the quasi-random numbers generated
extern void genWardSamples(int smpls, float alphax, float alphay);
