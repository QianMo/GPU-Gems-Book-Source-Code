#ifndef __TREE_H
#define __TREE_H

#include "Cfg.h"

///////////////////////////////////////////////////////////////////////////////
// Branch description
// Array of branch descriptions forms a tree description
//
struct BranchDesc
{
	// cone geometry
	float radiusFrom;
	float radiusTo;
	float length;

	// bending exponent
	float bendExp;

	// number of child branches along given branch
	size_t childrenCount;

	// defines 'weakening' of children
	float childModifier;

	// positions range for children
	// valid range [0..1]
	float childrenPosFrom;
	float childrenPosTo;

	// defines orientations of children
	float childrenPosAngle;

	// child relative angle from parent
	float childrenDirAngle;
};

#endif