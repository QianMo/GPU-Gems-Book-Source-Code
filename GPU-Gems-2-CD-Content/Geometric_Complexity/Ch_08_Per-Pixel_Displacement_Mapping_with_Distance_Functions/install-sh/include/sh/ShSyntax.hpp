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
#ifndef SHSYNTAX_HPP
#define SHSYNTAX_HPP

#include "ShDllExport.hpp"
#include "ShProgram.hpp"
#include "ShUtility.hpp"

/** \file ShSyntax.hpp
 * Sh syntactical definitions.
 */

/// @internal
//@{
#define SH_PUSH_ARG_QUEUE ::SH::shPushArgQueue()
#define SH_PUSH_ARG ::SH::shPushArg()
#define SH_PROCESS_ARG(arg, internal_cond) ::SH::shProcessArg(arg, internal_cond)
//@}

/// @name Shader definitions
//@{
/** \def SH_BEGIN_PROGRAM(target)
 * Begin a new program block.
 * Nesting programs is not allowed.
 * @retval ShProgram A reference to the new program.
 * @see SH_END_PROGRAM
 */
#define SH_BEGIN_PROGRAM(target) ::SH::shBeginShader(target);
/** \def SH_END_PROGRAM
 * End the current program definition.
 *
 * If there is a backend and the program has a specified target
 * (e.g. vertex or fragment), this will also compile the program.
 *
 * @see SH_BEGIN_PROGRAM
 */
#define SH_END_PROGRAM         ::SH::shEndShader();
//@}

#define SH_END                 SH_END_PROGRAM

#define SH_BEGIN_VERTEX_PROGRAM ::SH::shBeginShader("gpu:vertex");
#define SH_BEGIN_FRAGMENT_PROGRAM ::SH::shBeginShader("gpu:fragment");

///@name If statements
//@{
/** \def SH_IF(cond)
 * Begin an if statement.
 * @see SH_ELSE
 * @see SH_ENDIF
 */
#define SH_IF(cond) { \
  bool sh__internal_cond; \
  ::SH::shIf(SH_PUSH_ARG_QUEUE && SH_PUSH_ARG && SH_PROCESS_ARG(cond, &sh__internal_cond)); \
  if (ShContext::current()->parsing() || sh__internal_cond) {{
/** \def SH_ELSE
 * Indicate the start of the else-block of an if statement.
 * @see SH_IF
 * @see SH_ENDIF
 */
#define SH_ELSE  \
  }} \
  ::SH::shElse(); \
  if (ShContext::current()->parsing() || !sh__internal_cond) {{
/** \def SH_ENDIF
 * Indicate the end of an if-statement.
 * @see SH_IF
 * @see SH_ELSE
 */
#define SH_ENDIF \
  }} \
  ::SH::shEndIf(); \
}
//@}

/// @name While loops
//@{
/** \def SH_WHILE(cond)
 * Begin a while statement, iterating as long as \a cond is satisfied.
 * @see SH_ENDWHILE
 */
#define SH_WHILE(cond) { \
  bool sh__internal_cond; \
  ::SH::shWhile(SH_PUSH_ARG_QUEUE && SH_PUSH_ARG && SH_PROCESS_ARG(cond, &sh__internal_cond)); \
  bool sh__internal_firsttime = true; \
  while ((sh__internal_firsttime && (ShContext::current()->parsing() || sh__internal_cond)) \
         || (!ShContext::current()->parsing() && ::SH::shEvaluateCondition(cond))) {{{
/** \def SH_ENDWHILE
 * Indicate the end of a while-statement.
 * @see SH_WHILE
 */
#define SH_ENDWHILE \
    sh__internal_firsttime = false; \
  }}} \
  ::SH::shEndWhile(); \
}
//@}

/// @name Do-until loops
//@{
/** \def SH_DO
 * Indicate the start of a do-until statement.
 * @see SH_UNTIL
 */
#define SH_DO       { \
  ::SH::shDo(); \
  do {{{{
/** \def SH_UNTIL(cond)
 * End a do-until statement, iterating as long as \a cond is satisfied.
 * @see SH_DO
 */
#define SH_UNTIL(cond) \
  }}}} while (!ShContext::current()->parsing() && ::SH::shEvaluateCondition(cond)); \
  if (ShContext::current()->parsing()) ::SH::shUntil(SH_PUSH_ARG_QUEUE && SH_PUSH_ARG && SH_PROCESS_ARG(cond, 0)); }
//@}

/// @name For loops
//@{
/** \def SH_FOR(init, cond, update)
 * Begin a for statement.
 * The semantics are mostly the same as for C++ for statements, but
 * unlike in C++ declaring variables in \a init part is not allowed.
 * @see SH_ENDFOR
 */
// TODO: It is possible to make declaring variables in init work. do so.
#define SH_FOR(init,cond,update) { \
  if (ShContext::current()->parsing()) \
    ::SH::shFor(SH_PUSH_ARG_QUEUE \
             && SH_PUSH_ARG && SH_PROCESS_ARG(init, 0) \
             && SH_PUSH_ARG && SH_PROCESS_ARG(cond, 0) \
             && SH_PUSH_ARG && SH_PROCESS_ARG(update, 0)); \
  if (!ShContext::current()->parsing()) init; \
  bool sh__internal_first_time = true; \
  while (1) {{{{{ \
    if (!ShContext::current()->parsing()) { \
      if (!sh__internal_first_time) { update; } \
      else { sh__internal_first_time = false; } \
      if (!shEvaluateCondition(cond)) break; \
    }                                                                                  
/** \def SH_ENDFOR
 * Indicate the end of a for statement
 * @see SH_FOR
 */
#define SH_ENDFOR \
    if (ShContext::current()->parsing()) break; \
  }}}}} \
  ::SH::shEndFor(); \
}
//@}

/// @name Loop flow control
//@{
/** \def SH_BREAK
 * Break out of a loop, terminating the loop.
 * @see SH_CONTINUE
 * @see SH_WHILE
 * @see SH_DO
 * @see SH_FOR
 */
#define SH_BREAK    if (!ShContext::current()->parsing()) { break; } else { ::SH::shBreak(); }
/** \def SH_CONTINUE
 * Break out of a loop, continuing with the next iteration.
 * @see SH_BREAK
 * @see SH_WHILE
 * @see SH_DO
 * @see SH_FOR
 */
#define SH_CONTINUE if (!ShContext::current()->parsing()) { continue; } else { ::SH::shBreak(); }
//@}

/// @name Named Declaration macros 
//@{
/** \def SH_NAME
 * Set the name of a variable to be its C++ name.
 *
 * @see SH_DECL
 * @see SH_NAMEDECL
 */
#define SH_NAME(var) do { var.name( # var); } while (0)

/** \def SH_DECL
 * Declare variable with the same name as var.
 * Usage:
 *  Sh_Some_Type SH_DECL(var);
 * or 
 *  Sh_Some_TYPE SH_DECL(var) = initial_value;
 *
 * @see SH_NAMEDDECL
 */
#define SH_DECL(var) var; var.name( # var ); ::SH::ShIgnore() & var

/** \def SH_NAMEDECL
 * Declare variable with the given name 
 * Usage:
 *  Sh_Some_Type SH_NAMEDECL(var, "name");
 * or Sh_Some_Type SH_NAMEDECL(var, "name") = initial_value;
 *
 * @see SH_DECL
 */
#define SH_NAMEDECL(var, varName) var; var.name( varName ); ::SH::ShIgnore() & var
//@}

namespace SH {

/// \internal
SH_DLLEXPORT
bool shProcessArg(const ShVariable& arg, bool* internal_cond);

/// \internal
SH_DLLEXPORT
bool shPushArg();

/// \internal
SH_DLLEXPORT
bool shPushArgQueue();

/// \internal
SH_DLLEXPORT
bool shEvaluateCondition(const ShVariable& arg);

/// \internal
SH_DLLEXPORT
ShProgram shBeginShader(const std::string& kind = "");
/// \internal
SH_DLLEXPORT
void shEndShader();

/// Force (re)compilation of a program under its default target.
SH_DLLEXPORT
void shCompile(ShProgram& prg);
/// Force (re)compilation of a program for a given target.
SH_DLLEXPORT
void shCompile(ShProgram& prg, const std::string& target);

/// \deprecated Use shCompile() instead
SH_DLLEXPORT
void shCompileShader(ShProgram& prg);
/// \deprecated Use shCompile() instead
SH_DLLEXPORT
void shCompileShader(ShProgram& prg, const std::string& target);

/// Bind a program using the program's default target
SH_DLLEXPORT
void shBind(ShProgram& prg);
/// Bind a program with the given target.
SH_DLLEXPORT
void shBind(const std::string& target, ShProgram& shader);

/// \deprecated Use shBind() instead.
SH_DLLEXPORT
void shBindShader(ShProgram& shader);
/// \deprecated Use shBind() instead.
SH_DLLEXPORT
void shBindShader(const std::string& target, ShProgram& shader);

/// Switch to a particular backend
SH_DLLEXPORT
bool shSetBackend(const std::string& name);


/** \brief SH Initialization Function.
 *
 * The function needs to be called prior to the use of any other SH functions. Additionally,
 * in Windows, this function needs to be called after a OpenGL context/window has been created.
 */
SH_DLLEXPORT
void shInit();

/// \internal
SH_DLLEXPORT
void shIf(bool);
/// \internal
SH_DLLEXPORT
void shElse();
/// \internal
SH_DLLEXPORT
void shEndIf();

/// \internal
SH_DLLEXPORT
void shWhile(bool);
/// \internal
SH_DLLEXPORT
void shEndWhile();

/// \internal
SH_DLLEXPORT
void shDo();
/// \internal
SH_DLLEXPORT
void shUntil(bool);

/// \internal
SH_DLLEXPORT
void shFor(bool);
/// \internal
SH_DLLEXPORT
void shEndFor();

/// \internal
SH_DLLEXPORT
void shBreak();
/// \internal
SH_DLLEXPORT
void shContinue();

}

#endif
