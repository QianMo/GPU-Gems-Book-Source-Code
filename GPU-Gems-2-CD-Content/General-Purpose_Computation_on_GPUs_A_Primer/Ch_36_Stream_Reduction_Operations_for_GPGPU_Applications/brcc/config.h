/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

    CTool Library
    Copyright (C) 1998-2001	Shaun Flisakowski

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 1, or (at your option)
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

    o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */

#ifndef CONFIG_H
#define CONFIG_H

/**************************************************************************/

#ifdef _WIN32
#define strcasecmp      _stricmp
#endif

#ifdef __CYGWIN__
#define WINDOWS
#endif

/**************************************************************************/

#define EXTERN  extern

/**************************************************************************/
/*
    The following macro helps define argument lists for fns: the arg lists are
    eaten up when not allowed (as in C).  e.g. extern int foo ARGS((int, int));
*/

#if defined(__STDC__) || defined(__cplusplus)
#    define ARGS(args) args
#else
#    define ARGS(args) ()
#endif


#if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)
#	define	__PRETTY_FUNCTION__		"<Pretty Function Name (gcc only)>"
#endif

/**************************************************************************/

// For the gcc extension \e in strings.
#define    ESC_VAL    ('\033')

/**************************************************************************/

#endif /* CONFIG_H */
