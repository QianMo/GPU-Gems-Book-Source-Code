# Microsoft Developer Studio Project File - Name="flo" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=flo - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "flo.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "flo.mak" CFG="flo - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "flo - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "flo - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "flo - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /MD /W3 /GX /O2 /I "./" /I "../" /I "../rendertexture" /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 cg.lib cggl.lib glew32.lib glut32.lib glu32.lib opengl32.lib /nologo /subsystem:console /machine:I386

!ELSEIF  "$(CFG)" == "flo - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Gm /GX /ZI /Od /I "./" /I "../" /I "../rendertexture" /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 cg.lib cggl.lib glew32.lib glut32.lib glu32.lib opengl32.lib /nologo /subsystem:console /map /debug /machine:I386 /pdbtype:sept

!ENDIF 

# Begin Target

# Name "flo - Win32 Release"
# Name "flo - Win32 Debug"
# Begin Group "Cg Fragment Programs"

# PROP Default_Filter ".cg"
# Begin Source File

SOURCE=.\flo.cg
# End Source File
# Begin Source File

SOURCE=.\floUtil.cg
# End Source File
# End Group
# Begin Group "Core Classes"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\flo.cpp
# End Source File
# Begin Source File

SOURCE=.\flo.h
# End Source File
# Begin Source File

SOURCE=.\main.cpp
# End Source File
# Begin Source File

SOURCE=.\readme.txt
# End Source File
# End Group
# Begin Group "Support Classes"

# PROP Default_Filter ""
# Begin Source File

SOURCE=..\RenderTexture\RenderTexture.cpp
# End Source File
# Begin Source File

SOURCE=..\RenderTexture\RenderTexture.h
# End Source File
# Begin Source File

SOURCE=..\geep\slabop.h
# End Source File
# Begin Source File

SOURCE=..\geep\SlabOpCgGL.cpp
# End Source File
# Begin Source File

SOURCE=..\geep\SlabOpCgGL.h
# End Source File
# Begin Source File

SOURCE=..\geep\SlabOpGL.h
# End Source File
# Begin Source File

SOURCE=.\stopwatch.cpp
# End Source File
# Begin Source File

SOURCE=.\stopwatch.hpp
# End Source File
# Begin Source File

SOURCE=.\tga.cpp
# End Source File
# Begin Source File

SOURCE=.\tga.hpp
# End Source File
# End Group
# End Target
# End Project
