dnl
dnl AM_PATH_CG([ACTION-IF_FOUND [, ACTION-IF-NOT-FOUND]]).
dnl
dnl Test for Cg libraries, and define CG_CFLAGS and CG_LIBS.
dnl 
dnl Shamelessly stolen from ogg.m4
dnl

AC_DEFUN(AM_PATH_CG,
[dnl 
dnl Get the cflags and libraries
dnl

AC_ARG_WITH(cg,[  --with-cg=PFX             Prefix where Cg is installed (optional)], cg_prefix="$withval", cg_prefix="")

AC_ARG_WITH(cg-libraries,[  --with-cg-libraries=DIR   Directory where Cg libraries are installed (optional)], cg_libraries="$withval", cg_libraries="")

AC_ARG_WITH(cg-includes,[  --with-cg-includes=DIR    Directory where Cg header files are installed (optional)], cg_includes="$withval", cg_includes="")

AC_ARG_ENABLE(cgtest, [  --disable-cgtest       Do not try to compile and run a test Cg program],, enable_cgtest=yes)

  if test "x$cg_libraries" != "x" ; then
    CG_LIBS="-L$cg_libraries"
  elif test "x$cg_prefix" != "x" ; then
    CG_LIBS="-L$cg_prefix/lib"
  elif test "x$prefix" != "xNONE" ; then
    CG_LIBS="-L$prefix/lib"
  fi

  CG_LIBS="$CG_LIBS -lGL -lCg -lCgGL"

  if test "x$cg_includes" != "x" ; then
    CG_CFLAGS="-I$cg_includes"
  elif test "x$cg_prefix" != "x" ; then
    CG_CFLAGS="-I$cg_prefix/include"
  elif test "$prefix" != "xNONE"; then
    CG_CFLAGS="-I$prefix/include"
  fi

  AC_MSG_CHECKING(for Cg)
  no_cg=""


  if test "x$enable_cgtest" = "xyes" ; then
    ac_save_CFLAGS="$CFLAGS"
    ac_save_CXXFLAGS="$CXXFLAGS"
    ac_save_LIBS="$LIBS"
    CFLAGS="$CFLAGS $CG_CFLAGS"
    CXXFLAGS="$CXXFLAGS $CG_CFLAGS"
    LIBS="$LIBS $CG_LIBS"
dnl
dnl Now check if the installed Cg is sufficiently new.
dnl
      rm -f conf.cgtest
      AC_TRY_RUN([
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Cg/cg.h>

int main ()
{
  system("touch conf.cgtest");
  return 0;
}

],, no_cg=yes,[echo $ac_n "cross compiling; assumed OK... $ac_c"])
       CFLAGS="$ac_save_CFLAGS"
       LIBS="$ac_save_LIBS"
  fi

  if test "x$no_cg" = "x" ; then
     AC_MSG_RESULT(yes)
     ifelse([$1], , :, [$1])     
  else
     AC_MSG_RESULT(no)
     if test -f conf.cgtest ; then
       :
     else
       echo "*** Could not run Cg test program, checking why..."
       CFLAGS="$CFLAGS $CG_CFLAGS"
       LIBS="$LIBS $CG_LIBS"
       AC_TRY_LINK([
#include <Cg/cg.h>
],     [ return 0; ],
       [ echo "*** The test program compiled, but did not run. This usually means"
       echo "*** that the run-time linker is not finding Cg or finding the wrong"
       echo "*** version of Cg. If it is not finding Cg, you'll need to set your"
       echo "*** LD_LIBRARY_PATH environment variable, or edit /etc/ld.so.conf to point"
       echo "*** to the installed location  Also, make sure you have run ldconfig if that"
       echo "*** is required on your system"
       echo "***"
       echo "*** If you have an old version installed, it is best to remove it, although"
       echo "*** you may also be able to get things to work by modifying LD_LIBRARY_PATH"],
       [ echo "*** The test program failed to compile or link. See the file config.log for the"
       echo "*** exact error that occured. This usually means Cg was incorrectly installed"
       echo "*** or that you have moved Cg since it was installed." ])
       CFLAGS="$ac_save_CFLAGS"
       LIBS="$ac_save_LIBS"
     fi
     CG_CFLAGS=""
     CG_LIBS=""
     ifelse([$2], , :, [$2])
  fi
  AC_SUBST(CG_CFLAGS)
  AC_SUBST(CG_LIBS)
  rm -f conf.cgtest
])
