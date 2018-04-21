# Microsoft Developer Studio Project File - Name="TestRenderDepthTextureCg" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=TestRenderDepthTextureCg - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "TestRenderDepthTextureCg.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "TestRenderDepthTextureCg.mak" CFG="TestRenderDepthTextureCg - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "TestRenderDepthTextureCg - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "TestRenderDepthTextureCg - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "TestRenderDepthTextureCg - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "TestRenderDepthTextureCg___Win32_Release"
# PROP BASE Intermediate_Dir "TestRenderDepthTextureCg___Win32_Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 cg.lib cgGL.lib glew32.lib opengl32.lib glu32.lib glut32.lib /nologo /subsystem:console /machine:I386

!ELSEIF  "$(CFG)" == "TestRenderDepthTextureCg - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "TestRenderDepthTextureCg___Win32_Debug"
# PROP BASE Intermediate_Dir "TestRenderDepthTextureCg___Win32_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 cg.lib cgGL.lib glew32.lib opengl32.lib glu32.lib glut32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept

!ENDIF 

# Begin Target

# Name "TestRenderDepthTextureCg - Win32 Release"
# Name "TestRenderDepthTextureCg - Win32 Debug"
# Begin Source File

SOURCE=.\RenderTexture.cpp
# End Source File
# Begin Source File

SOURCE=.\RenderTexture.h
# End Source File
# Begin Source File

SOURCE=.\test.cg

!IF  "$(CFG)" == "TestRenderDepthTextureCg - Win32 Release"

# PROP Ignore_Default_Tool 1
# Begin Custom Build - Performing Cg compile on $(InputPath)
IntDir=.\Release
InputPath=.\test.cg
InputName=test

"$(IntDir)/$(InputName).fp" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"
	$(CG_COMPILER_EXE) $(InputPath) -o $(IntDir)/$(InputName).fp -profile arbfp1

# End Custom Build

!ELSEIF  "$(CFG)" == "TestRenderDepthTextureCg - Win32 Debug"

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\TestRenderDepthTextureCg.cpp
# End Source File
# Begin Source File

SOURCE=.\texture2D.cg

!IF  "$(CFG)" == "TestRenderDepthTextureCg - Win32 Release"

# PROP Ignore_Default_Tool 1
# Begin Custom Build - Performing Cg compile on $(InputPath)
IntDir=.\Release
InputPath=.\texture2D.cg
InputName=texture2D

"$(IntDir)/$(InputName).fp" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"
	$(CG_COMPILER_EXE) $(InputPath) -o $(IntDir)/$(InputName).fp -profile arbfp1

# End Custom Build

!ELSEIF  "$(CFG)" == "TestRenderDepthTextureCg - Win32 Debug"

# PROP Ignore_Default_Tool 1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\textureRECT.cg

!IF  "$(CFG)" == "TestRenderDepthTextureCg - Win32 Release"

# PROP Ignore_Default_Tool 1
# Begin Custom Build - Performing Cg compile on $(InputPath)
IntDir=.\Release
InputPath=.\textureRECT.cg
InputName=textureRECT

"$(IntDir)/$(InputName).fp" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"
	$(CG_COMPILER_EXE) $(InputPath) -o $(IntDir)/$(InputName).fp -profile arbfp1

# End Custom Build

!ELSEIF  "$(CFG)" == "TestRenderDepthTextureCg - Win32 Debug"

# PROP Ignore_Default_Tool 1

!ENDIF 

# End Source File
# End Target
# End Project
