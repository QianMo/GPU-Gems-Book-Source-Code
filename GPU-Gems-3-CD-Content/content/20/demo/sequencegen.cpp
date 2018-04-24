//
// seqencegen.cpp
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

// The following uses the PDSampler code based provided by Daniel Dunbar
// and the radiacal inverse code by Matt Pharr and Greg Humphreys 
// to generate a variety of different quasi-random number sequences.

#define _USE_MATH_DEFINES
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <PDSampling.h>
#include "sequencegen.h"

using namespace std;

static const float minrnd=0.000001;
static const float maxrnd=1-minrnd;

// globals
float *randnum=NULL, *randcos=NULL, *randsin=NULL, *randlog=NULL;
int randBufferSize=0;

// TODO: CLEAN UP THIS MEMORY
float *lSmpls=NULL, *lScales=NULL, *wSmpls=NULL, *wScales=NULL;
int lSmplSz=0, wSmplSz=0;

SequenceType currSequenceType = FOLDED_HAMMERSLEY;

void genHaltonSequence(int n);
void genFoldedHaltonSequence(int n);
void genHammersleySequence(int n);
void genFoldedHammersleySequence(int n);
void genPossionDiskSequence(int n);
void genBestCandidateSequence(int n);
void genPenroseSequence(int n);

// Inversion code courtesy of Matt Pharr and Greg Humphries
inline double FoldedRadicalInverse(int n, int base) {
	double val = 0;
	double invBase = 1.f/base, invBi = invBase;
	int modOffset = 0;
	while (val + base * invBi != val) {
		// Compute next digit of folded radical inverse
		int digit = ((n+modOffset) % base);
		val += digit * invBi;
		n /= base;
		invBi *= invBase;
		++modOffset;
	}
	return val;
}

inline double RadicalInverse(int n, int base) {
	double val = 0;
	double invBase = 1. / base, invBi = invBase;
	while (n > 0) {
		// Compute next digit of radical inverse
		int d_i = (n % base);
		val += d_i * invBi;
		n /= base;
		invBi *= invBase;
	}
	return val;
}

void allocateMemory(int n) {
	if (randBufferSize < n) {
		randnum	= (float*) realloc(randnum, sizeof(float)*n*4);
		randcos = &randnum[n];
		randsin = &randcos[n];
		randlog = &randsin[n];
		randBufferSize = n;
	}
}

void genSequence(int n) {
	switch (currSequenceType) {
		case HALTON:			genHaltonSequence(n);			break;
		case FOLDED_HALTON:		genFoldedHaltonSequence(n);		break;
		case HAMMERSLEY:		genHammersleySequence(n);		break;
		case FOLDED_HAMMERSLEY:	genFoldedHammersleySequence(n);	break;
		case POSSION_DISK:		genPossionDiskSequence(n);		break;
		case BEST_CANDIDATE:	genBestCandidateSequence(n);	break;
		case PENROSE:			genPenroseSequence(n);			break;
	}
}

void genHaltonSequence(int n) {
	allocateMemory(n);
	for (int i=0; i < n; i++) {
		randnum[i] = (float) RadicalInverse(i,2);
		if(randnum[i]<minrnd) randnum[i]=minrnd;
		if(randnum[i]>maxrnd) randnum[i]=maxrnd;
		randlog[i] = logf(randnum[i]);

		double e2 = 2.*M_PI*RadicalInverse(i,3);
		randcos[i] = (float) cos(e2);
		randsin[i] = (float) sin(e2);		
	}
}

void genFoldedHaltonSequence(int n) {
	allocateMemory(n);
	for (int i=0; i < n; i++) {
		randnum[i] = (float) FoldedRadicalInverse(i,2);
		if(randnum[i]<minrnd) randnum[i]=minrnd;
		if(randnum[i]>maxrnd) randnum[i]=maxrnd;
		randlog[i] = logf(randnum[i]);

		double e2 = 2.*M_PI*FoldedRadicalInverse(i,3);
		randcos[i] = (float) cos(e2);
		randsin[i] = (float) sin(e2);		
	}
}

void genHammersleySequence(int n) {
	allocateMemory(n);
	for (int i=0; i < n; i++) {
		randnum[i] = ((float) i)/((float) n);
		if(randnum[i]<minrnd) randnum[i]=minrnd;
		if(randnum[i]>maxrnd) randnum[i]=maxrnd;
		randlog[i] = logf(randnum[i]);

		double e2 = 2.*M_PI*RadicalInverse(i,2);
		randcos[i] = (float) cos(e2);
		randsin[i] = (float) sin(e2);		
	}
}

void genFoldedHammersleySequence(int n) {
	allocateMemory(n);
	for (int i=0; i < n; i++) {
		randnum[i] = ((float) i)/((float) n);
		if(randnum[i]<minrnd) randnum[i]=minrnd;
		if(randnum[i]>maxrnd) randnum[i]=maxrnd;
		randlog[i] = logf(randnum[i]);

		double e2 = 2.*M_PI*FoldedRadicalInverse(i,2);
		randcos[i] = (float) cos(e2);
		randsin[i] = (float) sin(e2);		
	}
}

void genPossionDiskSequence(int n) {
	float radius = 0.9f/sqrtf((float) n);
	radius = min(radius, 0.2f);
	PDSampler *sampler = new PureSampler(radius);

	sampler->complete();

	while ((sampler->points.size()-n) > 1000000)
		sampler->complete();

	for (int i=0; i < n; i++) {
		randnum[i] = sampler->points[i].x/2.f+0.5f;
		if(randnum[i]<minrnd) randnum[i]=minrnd;
		if(randnum[i]>maxrnd) randnum[i]=maxrnd;
		randlog[i] = logf(randnum[i]);

		float e2 = 2.f*((float) M_PI)*(sampler->points[i].y/2.f+0.5f);
		randcos[i] = (float) cosf(e2);
		randsin[i] = (float) sinf(e2);
	}

	delete sampler;
}

void genBestCandidateSequence(int n) {
	float radius = 0.75f/sqrtf((float) n);
	radius = min(radius, 0.2f);
	PDSampler *sampler = new BestCandidate(radius, true, 1);

	sampler->complete();

	while ((sampler->points.size()-n) > 1000000)
		sampler->complete();

	for (int i=0; i < n; i++) {
		randnum[i] = sampler->points[i].x/2.f+0.5f;
		if(randnum[i]<minrnd) randnum[i]=minrnd;
		if(randnum[i]>maxrnd) randnum[i]=maxrnd;
		randlog[i] = logf(randnum[i]);

		float e2 = 2.f*((float) M_PI)*(sampler->points[i].y/2.f+0.5f);
		randcos[i] = cosf(e2);
		randsin[i] = sinf(e2);
	}

	delete sampler;
}

void genPenroseSequence(int n) {
	float radius = 0.9f/sqrtf((float) n);
	radius = min(radius, 0.2f);
	PDSampler *sampler = new BestCandidate(radius, true, 1);

	sampler->complete();

	while ((sampler->points.size()-n) > 1000000)
		sampler->complete();

	for (int i=0; i < n; i++) {
		randnum[i] = sampler->points[i].x/2.f+0.5f;
		if(randnum[i]<minrnd) randnum[i]=minrnd;
		if(randnum[i]>maxrnd) randnum[i]=maxrnd;
		randlog[i] = logf(randnum[i]);

		float e2 = 2.f*((float) M_PI)*(sampler->points[i].y/2.f+0.5f);
		randcos[i] = cosf(e2);
		randsin[i] = sinf(e2);
	}

	delete sampler;
}

// precompute the samples from the quasi-random values
void genLafortuneSamples(int smpls, float cxy, float cz, float n) {
	if (smpls > lSmplSz) {
		lSmpls = (float*) realloc(lSmpls, smpls*sizeof(float)*5);
		lScales = &lSmpls[smpls*4];
		lSmplSz = smpls;
	}

	// format of the data
	// - u_x, u_y, u_z, biasCmp
	// - normalize*BRDF/PDF (partial)

	// Log[2, I*I/N/(2Pi)]*0.5 = Log[2, 512*512/40/(2*Pi)]*0.5 = 5.01329+1
	float lodPreComp = logf(512.f*512.f/((float) smpls)/(2.f*((float) M_PI)))/((float) M_LN2)*0.5f+1.f;

	for (int i=0; i < smpls; i++) {
		float costheta = powf(randnum[i], 1.f/(n+1.f));
		float sintheta = sqrtf(1.f - costheta*costheta);
		
		lSmpls[i*4+0] = randcos[i]*sintheta;
		lSmpls[i*4+1] = randsin[i]*sintheta;
		lSmpls[i*4+2] = costheta;
		
		float pdf = (n+1)*powf(costheta, n)/(((float) M_PI)*2.f);
		
		
		lSmpls[i*4+3] = max(0.f, lodPreComp - logf(pdf)/((float) M_LN2)*0.5f);
		lScales[i] = (n+2)*powf(costheta, n)/pdf;
	}

	cgGLSetParameterArray4f(lSmplsParam, 0, smpls, lSmpls);
	cgGLSetParameterArray1f(lScalesParam, 0, smpls, lScales);
}

void genWardSamples(int smpls, float alphax, float alphay) {
	if (smpls > wSmplSz) {
		wSmpls = (float*) realloc(wSmpls, smpls*sizeof(float)*5);
		wScales = &wSmpls[smpls*4];
		wSmplSz = smpls;
	}

	for (int i=0; i < smpls; i++) {
		float cosp = randcos[i] * alphax;
		float sinp = randsin[i] * alphay;

		float d = 1.f/sqrtf(cosp*cosp + sinp*sinp);
		cosp *= d;
		sinp *= d;

		d = -randlog[i] /( cosp*cosp/alphax/alphax + sinp*sinp/alphay/alphay);
		float hz = sqrtf(1.f/(d+1.f));
		float sint = sqrt(d)*hz;
		
		wSmpls[i*4+0] = sint*cosp;
		wSmpls[i*4+1] = sint*sinp;
		wSmpls[i*4+2] = hz;
		
		// compute the partial PDF (full pdf requires viewing direction)
		float exps = wSmpls[i*4+0]*wSmpls[i*4+0]/alphax/alphax + wSmpls[i*4+1]*wSmpls[i*4+1]/alphay/alphay;
		exps /= hz*hz;

		float norm = 1.f/(4.f*((float) M_PI)*alphax*alphay);

		float pdf = norm/(hz*hz*hz)*expf(-exps); // /dot_hv (done in shader)
		wSmpls[i*4+3] = pdf;

		wScales[i] = hz*hz*hz;

	}

	cgGLSetParameterArray4f(wSmplsParam, 0, samples, wSmpls);
	cgGLSetParameterArray1f(wScalesParam, 0, samples, wScales);

}