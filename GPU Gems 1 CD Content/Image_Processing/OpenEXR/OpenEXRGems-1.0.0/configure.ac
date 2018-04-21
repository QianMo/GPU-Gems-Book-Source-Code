dnl Process this file with autoconf to produce a configure script.

AC_INIT(exrplay/main.cpp)

dnl Checks for target
AC_CANONICAL_TARGET

OPENEXRGEMS_MAJOR_VERSION=1
OPENEXRGEMS_MINOR_VERSION=0
OPENEXRGEMS_MICRO_VERSION=0

OPENEXRGEMS_VERSION=$OPENEXRGEMS_MAJOR_VERSION.$OPENEXRGEMS_MINOR_VERSION.${OPENEXRGEMS_MICRO_VERSION}
AC_SUBST(OPENEXR_VERSION)

dnl GNU automake support.
AM_INIT_AUTOMAKE(OpenEXRGems, ${OPENEXRGEMS_VERSION})
AM_MAINTAINER_MODE

dnl Checks for programs.
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_CC
AC_PROG_LN_S
AC_PROG_LIBTOOL
AC_PROG_MAKE_SET

dnl
dnl check for OpenEXR
dnl
AM_PATH_OPENEXR(,
AC_MSG_ERROR([
***
*** Can't build OpenEXR Gems because they require the OpenEXR libraries
*** and include files.
]))

dnl
dnl check for Cg libraries and headers.
dnl
AM_PATH_CG(,
AC_MSG_ERROR([
*** Can't build OpenEXR Gems because they require the Cg libraries and
*** include files.
]))

dnl
dnl Optimization flags
dnl

AC_MSG_CHECKING(for compiler optimization flags)
old_cflags=$CFLAGS
CFLAGS="$CFLAGS -O2 -funroll-loops -pipe"
AC_TRY_COMPILE([#include <stdio.h>],
[ printf ("hello, world"); ],
[ EXTRA_OPT_CFLAGS="-O2 -funroll-loops -pipe"],[ EXTRA_OPT_CFLAGS=""])
CFLAGS=$old_cflags
AC_MSG_RESULT([$EXTRA_OPT_CFLAGS])

AC_OUTPUT([
Makefile
Glut/Makefile
exrplay/Makefile
exrcomp/Makefile
])

