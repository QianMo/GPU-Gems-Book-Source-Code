/*
 * fxc.h --
 *
 *      Interface to fxc-- Used to call fxc.exe compiler
 */
#ifndef __FXC_H__
#define __FXC_H__

#include "codegen.h"

char *
compile_fxc (const char *name, 
             const char *shader, 
             CodeGenTarget target, 
             ShaderResourceUsage* outUsage = 0, 
             bool inValidate = true );

#endif
