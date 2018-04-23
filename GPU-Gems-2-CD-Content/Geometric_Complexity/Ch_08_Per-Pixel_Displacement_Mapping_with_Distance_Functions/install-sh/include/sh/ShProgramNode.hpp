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
#ifndef SHPROGRAMNODE_HPP
#define SHPROGRAMNODE_HPP

#include <list>
#include <map>
#include <utility>
#include <string>
#include "ShDllExport.hpp"
#include "ShRefCount.hpp"
#include "ShTokenizer.hpp"
#include "ShVariableNode.hpp"
#include "ShCtrlGraph.hpp"
#include "ShTextureNode.hpp"
#include "ShChannelNode.hpp"
#include "ShPaletteNode.hpp"
#include "ShMeta.hpp"

namespace SH {

class ShBackendCode;
class ShBackend;

/** A particular Sh program.
 */
class
SH_DLLEXPORT ShProgramNode : public virtual ShRefCountable, public virtual ShMeta {
public:
  ShProgramNode(const std::string& target);

  ~ShProgramNode();
  
  /// Forcefully compile this program for a particular backend, even if
  /// it has been compiled previously. Use code() to obtain the actual
  /// code.
  /// This operation will fail if this program does not have a
  /// particular target.
  void compile(const ShPointer<ShBackend>& backend);

  /// Forcefully compile this program for a particular backend, even if
  /// it has been compiled previously. Use code() to obtain the actual code.
  void compile(const std::string& target, const ShPointer<ShBackend>& backend);

  /// Describe the inputs and outputs of this program.
  std::string describe_interface() const;
  
  /// Obtain the code for currently active backend. 
  /// This operation will fail if this program does not have a
  /// particular target.
  ShPointer<ShBackendCode> code(); 
  
  /// Obtain the code for a particular backend. Generates it if necessary.
  /// This operation will fail if this program does not have a
  /// particular target.
  ShPointer<ShBackendCode> code(const ShPointer<ShBackend>& backend);

  /// Obtain the code for a particular backend. Generates it if necessary.
  ShPointer<ShBackendCode> code(const std::string& target, const ShPointer<ShBackend>& backend);

  /// Notify this program that a uniform variable has changed.
  void updateUniform(const ShVariableNodePtr& uniform);

  /// The tokenizer for this program's body. Used only during
  /// construction of the program (before parsing)
  ShTokenizer tokenizer;

  /// The control graph (the parsed form of the token
  /// list). Constructed during the parsing step, when shEndProgram()
  /// is called.
  ShPointer<ShCtrlGraph> ctrlGraph;

  /// Call after contructing the control graph [after optimization!]
  /// to make lists of all the variables used in the program.
  void collectVariables();

  typedef std::list<ShVariableNodePtr> VarList;
  typedef std::list<ShTextureNodePtr> TexList;
  typedef std::list<ShChannelNodePtr> ChannelList;
  typedef std::list<ShPaletteNodePtr> PaletteList;

  VarList::const_iterator inputs_begin() const;
  VarList::const_iterator inputs_end() const;
  VarList::const_iterator outputs_begin() const;
  VarList::const_iterator outputs_end() const;
  VarList::const_iterator temps_begin() const;
  VarList::const_iterator temps_end() const;
  VarList::const_iterator constants_begin() const;
  VarList::const_iterator constants_end() const;
  VarList::const_iterator uniforms_begin() const;
  VarList::const_iterator uniforms_end() const;

  TexList::const_iterator textures_begin() const;
  TexList::const_iterator textures_end() const;

  ChannelList::const_iterator channels_begin() const;
  ChannelList::const_iterator channels_end() const;

  PaletteList::const_iterator palettes_begin() const;
  PaletteList::const_iterator palettes_end() const;
  
  
  VarList inputs; ///< Input variables used in this program
  VarList outputs; ///< Output variables used in this program
  VarList temps; ///< Temporary variables used in this program
  VarList constants; ///< Constants used in this program
  VarList uniforms; ///< Uniform variables used in this program
  TexList textures; ///< Textures used in this program
  ChannelList channels; ///< Channels used in FETCH instructions in this program
  PaletteList palettes; ///< Palettes used in PAL instructions in this program

  /// Can be empty, if there is no target associated with this program.
  std::string target() const { return m_target; }

  /// It may be useful to change a program's target sometimes.
  std::string& target() { return m_target; }

  /// Make a copy of this program.
  ShPointer<ShProgramNode> clone() const;

  /// Print a description of a list of variables
  static std::ostream& print(std::ostream& out, const VarList& list);

  /// True if this program has been completed with SH_END.
  bool finished() const;

  /// @internal Set finished to true. Only shEndShader() needs to call this.
  void finish();

  /// @internal Indicate that we have been assigned to a uniform
  /// during construction. This is so we can call back that uniform
  /// when the program is finished constructing.
  void assign(const ShVariableNodePtr& var) const;
  
private:

  std::string m_target; ///< Can be empty, if there is no target associated with this program.

  void collectNodeVars(const ShPointer<ShCtrlGraphNode>& node);
  void collectVar(const ShVariableNodePtr& node);

  typedef std::map< std::pair< std::string, ShPointer<ShBackend> >,
                    ShPointer<ShBackendCode> > CodeMap;
  CodeMap m_code; ///< Compiled code is cached here.

  bool m_finished; ///< True if this program has been constructed
                   /// completely.

  mutable ShVariableNodePtr m_assigned_var;
};

typedef ShPointer<ShProgramNode> ShProgramNodePtr;
typedef ShPointer<const ShProgramNode> ShProgramNodeCPtr;

}

#endif

