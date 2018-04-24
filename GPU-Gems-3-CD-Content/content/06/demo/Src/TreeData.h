#ifndef __TREE_DATA_H
#define __TREE_DATA_H

#include "Cfg.h"
#include "Tree.h"
#include "Parameters.h"
#include "Utils.h"

///////////////////////////////////////////////////////////////////////////////
// Array of branch descriptions forms a tree description
//

static const BranchDesc WEAK_FIR_TREE_DESC[] = {
//    r0     r1       len    bendExp  childCount
//									           mod	  pFrom	 pTo      posAngle		   dirAngle
    { 0.7f,  0.05f,   50.0f, 1.0f,    20,      0.25f, 0.2f,  0.95f,   deg2rad(135.0f), deg2rad(120.0f) },
    { 0.25f, 0.0f,	  15.0f, 0.5f,    3,       0.5f,  0.2f,  0.8f,    deg2rad(120.0f), deg2rad(30.0f) },
    { 0.2f,	 0.0f,    7.0f,  0.75f,   0 }
};

static const BranchDesc STIFF_FIR_TREE_DESC[] = {
//    r0     r1       len    bendExp  childCount
//									           mod	  pFrom	 pTo      posAngle		   dirAngle
    { 0.7f,  0.05f,   50.0f, 0.5f,     20,     0.25f, 0.2f,  0.95f,   deg2rad(135.0f), deg2rad(120.0f) },
    { 0.25f, 0.0f,	  15.0f, 1.25f,    3,      0.5f,  0.2f,  0.8f,    deg2rad(120.0f), deg2rad(30.0f) },
    { 0.2f,	 0.0f,    7.0f,  1.5f,     0 }
};

static const BranchDesc BIRCH_TREE_DESC[] = {
//    r0     r1       len    bendExp  childCount
//									           mod	  pFrom	 pTo      posAngle		   dirAngle
    { 0.65f,  0.05f,  40.0f, 2.0f,    10,     0.5f,  0.35f,  0.9f,   deg2rad(90.0f),  deg2rad(45.0f) },
    { 0.3f,   0.0f,	  15.0f, 2.0f,    5,      0.0f,  0.2f,   0.8f,   deg2rad(120.0f), deg2rad(40.0f) },
    { 0.22f,  0.0f,   7.0f,  0.7f,    0 }
};

static const BranchDesc SMALL_BIRCH_TREE_DESC[] = {
//    r0     r1       len    bendExp   childCount
//									           mod	  pFrom	 pTo      posAngle		   dirAngle
    { 0.35f,  0.05f,  30.0f, 1.0f,     10,      0.5f,  0.35f,  0.9f,   deg2rad(90.0f),  deg2rad(45.0f) },
    { 0.15f,  0.0f,	  10.0f, 0.5f,     5,       0.0f,  0.2f,   0.8f,   deg2rad(120.0f), deg2rad(40.0f) },
    { 0.1f,   0.0f,   7.0f,  0.2f,     0 }
};



///////////////////////////////////////////////////////////////////////////////
// Simulation parameters for trees
//

static const SimulationParameters WEAK_FIR_TREE_PARAMS = {
    0.10f, 3.00f, 
    0.40f, 1.15f,
    
    {-0.51f, 0.00f, 0.25f },	// angleShift
    { 0.15f, 0.21f, 0.25f },	// amplitude
    { 1.00f, 1.00f, 0.75f },	// frequency

    {-0.75f, 0.00f, 0.00f },	// angleShift_
    { 0.07f, 0.38f, 0.00f },	// amplitude_
};

static const SimulationParameters STIFF_FIR_TREE_PARAMS = {
    0.03f, 3.50f,
    0.16f, 1.60f,
    
    {-0.41f, 0.00f, 0.06f },	// angleShift
    { 0.04f, 0.12f, 0.12f },	// amplitude
    { 1.00f, 1.00f, 0.80f },	// frequency
                            
    {-0.29f, 0.00f, 0.00f },	// angleShift_
    { 0.13f, 0.12f, 0.00f },	// amplitude_
};

static const SimulationParameters BIRCH_TREE_PARAMS = {
    0.10f, 3.50f,
    -0.09f,1.40f,
    
    { 0.18f, 0.00f, 0.34f },	// angleShift
    { 0.03f, 0.03f, 0.16f },	// amplitude
    { 2.00f, 1.30f, 1.20f },	// frequency
                            
    { 0.32f, 0.00f, 0.00f },	// angleShift_
    { 0.08f, 0.12f, 0.00f },	// amplitude_
};

static const SimulationParameters SMALL_BIRCH_TREE_PARAMS = {
    0.18f, 3.30f,
    0.32f, 1.75f,
    
    { 0.18f, 0.00f, 0.45f },	// angleShift
    { 0.10f, 0.15f, 0.21f },	// amplitude
    { 2.00f, 1.35f, 1.50f },	// frequency
                            
    { 0.62f, 0.00f, 0.00f },	// angleShift_
    { 0.05f, 0.20f, 0.00f },	// amplitude_
};

#endif