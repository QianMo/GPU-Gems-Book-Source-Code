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
#ifndef SHPROGRAM_HPP
#define SHPROGRAM_HPP

#include "ShDllExport.hpp"
#include "ShProgramNode.hpp"
#include "ShBackend.hpp"

namespace SH {

class ShStream;
template<typename T> class ShChannel;

/** Thin wrapper around ShProgramNode.
 */
class
SH_DLLEXPORT ShProgram : public ShMetaForwarder {
public:
  ShProgram();
  ShProgram(const ShProgram& other);
  ShProgram(const std::string& target);
  ShProgram(const ShProgramNodePtr& node);

  ShProgram& operator=(const ShProgram& other);
  
  /// Obtain the node which this ShProgram wraps
  ShProgramNodeCPtr node() const { return m_node; }

  /// Obtain the node which this ShProgram wraps
  ShProgramNodePtr node() { return m_node; }
  
  /// Forcefully compile this program for a particular backend, even if
  /// it has been compiled previously. Use code() to obtain the actual
  /// code.
  /// This operation will fail if this program does not have a
  /// particular target.
  void compile(const ShPointer<ShBackend>& backend) { m_node->compile(backend); }

  /// Forcefully compile this program for a particular backend, even if
  /// it has been compiled previously. Use code() to obtain the actual code.
  void compile(const std::string& target, const ShPointer<ShBackend>& backend)
  {
    m_node->compile(target, backend);
  }

  /// Obtain a listing of the inputs, outputs and uniforms used by
  /// this program.
  std::string describe_interface() const
  {
    return m_node->describe_interface();
  }
  
  /// Obtain the code for currently active backend. 
  /// This operation will fail if this program does not have a
  /// particular target.
  ShPointer<ShBackendCode> code() { return m_node->code(); }
  
  /// Obtain the code for a particular backend. Generates it if necessary.
  /// This operation will fail if this program does not have a
  /// particular target.
  ShPointer<ShBackendCode> code(const ShPointer<ShBackend>& backend) {
    return m_node->code(backend);
  }

  /// Obtain the code for a particular backend. Generates it if necessary.
  ShPointer<ShBackendCode> code(const std::string& target, const ShPointer<ShBackend>& backend)
  {
    return m_node->code(target, backend);
  }

  /// Notify this program that a uniform variable has changed.
  void updateUniform(const ShVariableNodePtr& uniform)
  {
    m_node->updateUniform(uniform);
  }

  std::string target() const { return m_node->target(); }

  /// It may be useful to change a program's target sometimes.
  std::string& target() { return m_node->target(); }

  ShProgramNode::VarList::const_iterator inputs_begin() const { return m_node->inputs_begin(); }
  ShProgramNode::VarList::const_iterator inputs_end() const { return m_node->inputs_end(); }
  ShProgramNode::VarList::const_iterator outputs_begin() const { return m_node->outputs_begin(); }
  ShProgramNode::VarList::const_iterator outputs_end() const { return m_node->outputs_end(); }
  ShProgramNode::VarList::const_iterator temps_begin() const { return m_node->temps_begin(); }
  ShProgramNode::VarList::const_iterator temps_end() const { return m_node->temps_end(); }
  ShProgramNode::VarList::const_iterator constants_begin() const { return m_node->constants_begin(); }
  ShProgramNode::VarList::const_iterator constants_end() const { return m_node->constants_end(); }
  ShProgramNode::VarList::const_iterator uniforms_begin() const { return m_node->uniforms_begin(); }
  ShProgramNode::VarList::const_iterator uniforms_end() const { return m_node->uniforms_end(); }
  ShProgramNode::TexList::const_iterator textures_begin() const { return m_node->textures_begin(); }
  ShProgramNode::TexList::const_iterator textures_end() const { return m_node->textures_end(); }
  ShProgramNode::ChannelList::const_iterator channels_begin() const { return m_node->channels_begin(); }
  ShProgramNode::ChannelList::const_iterator channels_end() const { return m_node->channels_end(); }
  ShProgramNode::PaletteList::const_iterator palettes_begin() const { return m_node->palettes_begin(); }
  ShProgramNode::PaletteList::const_iterator palettes_end() const { return m_node->palettes_end(); }

  // Call operators for channels and streams.
  // Equivalent to operator<< invocations.
  // Note that the template functions are implemented in
  // ShChannelImpl.hpp
  template<typename T0>
  ShProgram operator()(const ShChannel<T0>& t0) const;
  ShProgram operator()(const ShStream& s0) const;
  template<typename T0, typename T1>
  ShProgram operator()(const ShChannel<T0>& t0,
                        const ShChannel<T1>& t1) const;
  ShProgram operator()(const ShStream& s0,
                        const ShStream& s1) const;
  template<typename T0, typename T1, typename T2>
  ShProgram operator()(const ShChannel<T0>& t0,
                        const ShChannel<T1>& t1,
                        const ShChannel<T2>& t2) const;
  ShProgram operator()(const ShStream& s0,
                        const ShStream& s1,
                        const ShStream& s2) const;
  template<typename T0, typename T1, typename T2, typename T3>
  ShProgram operator()(const ShChannel<T0>& t0,
                        const ShChannel<T1>& t1,
                        const ShChannel<T2>& t2,
                        const ShChannel<T3>& t3) const;
  ShProgram operator()(const ShStream& s0,
                        const ShStream& s1,
                        const ShStream& s2,
                        const ShStream& s3) const;
  template<typename T0, typename T1, typename T2, typename T3,
           typename T4>
  ShProgram operator()(const ShChannel<T0>& t0,
                        const ShChannel<T1>& t1,
                        const ShChannel<T2>& t2,
                        const ShChannel<T3>& t3,
                        const ShChannel<T4>& t4) const;
  ShProgram operator()(const ShStream& s0,
                        const ShStream& s1,
                        const ShStream& s2,
                        const ShStream& s3,
                        const ShStream& s4) const;
  
private:

  ShProgramNodePtr m_node;
};

}

#endif

