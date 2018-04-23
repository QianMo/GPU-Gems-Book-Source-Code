/*
 * subprocess.h --
 *
 *      Interface to subprocess: a simple utility function for running a
 *      program and collecting its output.
 */
#ifndef __SUBPROCESS_H__
#define __SUBPROCESS_H__

extern char *
Subprocess_Run(char *argv[], char *input);

#endif
