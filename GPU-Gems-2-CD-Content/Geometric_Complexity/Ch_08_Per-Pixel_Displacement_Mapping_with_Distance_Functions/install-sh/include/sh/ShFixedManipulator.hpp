// Sh: A GPU metaprogramming language.
//
// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
// Project administrator: Michael D. McCool
// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
//          Michael D. McCool
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must
// not claim that you wrote the original software. If you use this
// software in a product, an acknowledgment in the product documentation
// would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must
// not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
// distribution.
//////////////////////////////////////////////////////////////////////////////
#ifndef SHFIXEDMANIPULATOR_HPP
#define SHFIXEDMANIPULATOR_HPP

#include <vector>
#include "ShDllExport.hpp"
#include "ShProgram.hpp"

namespace SH {

typedef ShProgramNode::VarList::const_iterator ShManipVarIterator;

/** A ShFixedManipulator is a ShProgram output manipulator. 
 * Fixed size manipulators can be combined with each other
 * to give "wider" fixed size manipulators that handle more 
 * ShProgram outputs.
 */
class
SH_DLLEXPORT ShFixedManipulatorNode: public ShRefCountable { 
  public:
    ShFixedManipulatorNode();
    virtual ~ShFixedManipulatorNode(); 

    /** Define the manipulator's behaviour on inputs.
     * Consumes a number of inputs in the variable list, changing finger.
     * This function must ensure that it doesn't pass the end iterator.
     * If it does, it must throw an ShAlgebraException
     */
    virtual ShProgram applyToInputs(ShManipVarIterator &finger, ShManipVarIterator end) const = 0;

    /** Defines the manipulator's behaviour on outputs. 
     * Consumes a number of outputs in the variable list, changing finger.
     * This function must ensure that it doesn't pass the end iterator.
     * If it does, it must throw an ShAlgebraException
     */
    virtual ShProgram applyToOutputs(ShManipVarIterator &finger, ShManipVarIterator end) const = 0;
};
typedef ShPointer<ShFixedManipulatorNode> ShFixedManipulator;

class
SH_DLLEXPORT ShKeepNode: public ShFixedManipulatorNode {
  public:
    ShKeepNode(int numChannels); 
    ShProgram applyToInputs(ShManipVarIterator &finger, ShManipVarIterator end) const;
    ShProgram applyToOutputs(ShManipVarIterator &finger, ShManipVarIterator end) const;

  private:
    int m_numChannels;
};
SH_DLLEXPORT
ShFixedManipulator shKeep(int numChannels = 1);

class
SH_DLLEXPORT ShLoseNode: public ShFixedManipulatorNode {
  public:
    ShLoseNode(int numChannels); 
    ShProgram applyToInputs(ShManipVarIterator &finger, ShManipVarIterator end) const;
    ShProgram applyToOutputs(ShManipVarIterator &finger, ShManipVarIterator end) const;

  private:
    int m_numChannels;
};
SH_DLLEXPORT
ShFixedManipulator shLose(int numChannels = 1);

class
SH_DLLEXPORT ShDupNode: public ShFixedManipulatorNode {
  public:
    ShDupNode(int numDups);
    ShProgram applyToInputs(ShManipVarIterator &finger, ShManipVarIterator end) const;
    ShProgram applyToOutputs(ShManipVarIterator &finger, ShManipVarIterator end) const;

  private:
    int m_numDups;
};
SH_DLLEXPORT
ShFixedManipulator shDup(int numDups = 2);

// TODO make class names less clunky
// This node can only be created by using the & operator with another fixed manipulator
//
// This is the only manip node that allows number of channels
// not to match when connected to a ShProgram. (extras are handled in the
// default connect way when inputs != outpus)
//
class
SH_DLLEXPORT ShProgramManipNode: public ShFixedManipulatorNode {
  public:
    ShProgram applyToInputs(ShManipVarIterator &finger, ShManipVarIterator end) const;
    ShProgram applyToOutputs(ShManipVarIterator &finger, ShManipVarIterator end) const;

  private:
    ShProgram p;

    ShProgramManipNode(const ShProgram &p);

  friend SH_DLLEXPORT
  ShFixedManipulator operator&(const ShFixedManipulator &m, const ShProgram &p );
  friend SH_DLLEXPORT
  ShFixedManipulator operator&( const ShProgram &p, const ShFixedManipulator &m);
};

// This node can only be created using the & operator with another fixed manipulator
class ShTreeManipNode: public ShFixedManipulatorNode {
  public:
    ShProgram applyToInputs(ShManipVarIterator &finger, ShManipVarIterator end) const;
    ShProgram applyToOutputs(ShManipVarIterator &finger, ShManipVarIterator end) const;
  private:
    ShFixedManipulator a, b;

    ShTreeManipNode(const ShFixedManipulator &a, const ShFixedManipulator &b); 

    friend SH_DLLEXPORT

    ShFixedManipulator operator&(const ShFixedManipulator &m, const ShFixedManipulator &n);
};

SH_DLLEXPORT
ShProgram operator<<(const ShProgram &p, const ShFixedManipulator &m); 
SH_DLLEXPORT
ShProgram operator<<(const ShFixedManipulator &m, const ShProgram &p);
SH_DLLEXPORT
ShFixedManipulator operator&(const ShFixedManipulator &m, 
			     const ShFixedManipulator &n);
SH_DLLEXPORT
ShFixedManipulator operator&(const ShFixedManipulator &m, 
			     const ShProgram &p );
SH_DLLEXPORT
ShFixedManipulator operator&(const ShProgram &p, 
			     const ShFixedManipulator &m);

}

#endif
