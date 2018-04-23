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

/*********************************************************************
/     Identify.h                                                     
/
/     Including this file will define HPUX, SUN, MIPS, or OTHER. 
/     (should probably contain VMS, SOLAR, and some others)
/
/ ********************************************************************/

#ifndef    IDENTIFY_H
#define    IDENTIFY_H

#include <sys/types.h>

#include "config.h"

/* HPUX -  */

#ifdef    _HPUX_SOURCE
#    define    HPUX
#endif

/*********************************************************************/

/* SUN  */

#ifdef    sun
#    define    SUN
#endif

/*********************************************************************/

/* MIPS -  */

#ifdef    __MIPSEL
#    define    MIPS
#endif

/*********************************************************************/

/* LINUX -  */

#ifdef    __linux
#    define    LINUX
#endif

/*********************************************************************/
/* DJGPP -  */

#ifdef __dj_include_sys_types_h_
#    define    DJGPP
#endif

/*********************************************************************/

/* MacOS X */
#ifdef __APPLE__
#	define	MACOSX
#endif

/*********************************************************************/

/* Catch all for others */

#if !defined(HPUX) && !defined(SUN) && !defined(MIPS)
#    if !defined(LINUX) && !defined(DJGPP)
#        define    OTHER
#    endif 
#endif 

/*********************************************************************/

#endif     /* IDENTIFY_H */
