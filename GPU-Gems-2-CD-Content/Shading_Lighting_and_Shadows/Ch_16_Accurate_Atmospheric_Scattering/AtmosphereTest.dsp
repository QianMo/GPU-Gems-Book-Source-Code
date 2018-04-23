# Microsoft Developer Studio Project File - Name="AtmosphereTest" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Application" 0x0101

CFG=AtmosphereTest - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "AtmosphereTest.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "AtmosphereTest.mak" CFG="AtmosphereTest - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "AtmosphereTest - Win32 Release" (based on "Win32 (x86) Application")
!MESSAGE "AtmosphereTest - Win32 Debug" (based on "Win32 (x86) Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "AtmosphereTest - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "AtmosphereTest___Win32_Release"
# PROP BASE Intermediate_Dir "AtmosphereTest___Win32_Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /Yu"Master.h" /FD /c
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib advapi32.lib winmm.lib opengl32.lib glu32.lib cg.lib cgGL.lib libjpeg.lib /nologo /subsystem:windows /machine:I386
# SUBTRACT LINK32 /debug

!ELSEIF  "$(CFG)" == "AtmosphereTest - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "AtmosphereTest___Win32_Debug"
# PROP BASE Intermediate_Dir "AtmosphereTest___Win32_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /Yu"Master.h" /FD /GZ /c
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib advapi32.lib winmm.lib opengl32.lib glu32.lib cg.lib cgGL.lib libjpeg.lib /nologo /subsystem:windows /debug /machine:I386 /pdbtype:sept

!ENDIF 

# Begin Target

# Name "AtmosphereTest - Win32 Release"
# Name "AtmosphereTest - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\AtmosphereTest.rc
# End Source File
# Begin Source File

SOURCE=.\GameApp.cpp
# End Source File
# Begin Source File

SOURCE=.\GameEngine.cpp
# End Source File
# Begin Source File

SOURCE=.\glprocs.c
# SUBTRACT CPP /YX /Yc /Yu
# End Source File
# Begin Source File

SOURCE=.\GLUtil.cpp
# End Source File
# Begin Source File

SOURCE=.\Master.cpp
# ADD CPP /Yc"Master.h"
# End Source File
# Begin Source File

SOURCE=.\Matrix.cpp
# End Source File
# Begin Source File

SOURCE=.\Noise.cpp
# End Source File
# Begin Source File

SOURCE=.\PBuffer.cpp
# End Source File
# Begin Source File

SOURCE=.\PixelBuffer.cpp
# End Source File
# Begin Source File

SOURCE=.\Texture.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\DateTime.h
# End Source File
# Begin Source File

SOURCE=.\Font.h
# End Source File
# Begin Source File

SOURCE=.\GameApp.h
# End Source File
# Begin Source File

SOURCE=.\GameEngine.h
# End Source File
# Begin Source File

SOURCE=.\glprocs.h
# End Source File
# Begin Source File

SOURCE=.\GLUtil.h
# End Source File
# Begin Source File

SOURCE=.\ListTemplates.h
# End Source File
# Begin Source File

SOURCE=.\Log.h
# End Source File
# Begin Source File

SOURCE=.\Master.h
# End Source File
# Begin Source File

SOURCE=.\Matrix.h
# End Source File
# Begin Source File

SOURCE=.\Noise.h
# End Source File
# Begin Source File

SOURCE=.\PBuffer.h
# End Source File
# Begin Source File

SOURCE=.\PixelBuffer.h
# End Source File
# Begin Source File

SOURCE=.\resource.h
# End Source File
# Begin Source File

SOURCE=.\Texture.h
# End Source File
# Begin Source File

SOURCE=.\WndClass.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# Begin Source File

SOURCE=.\icon1.ico
# End Source File
# End Group
# Begin Group "GLSL Shader Files"

# PROP Default_Filter "vert;frag"
# Begin Source File

SOURCE=.\GroundFromAtmosphere.frag
# End Source File
# Begin Source File

SOURCE=.\GroundFromAtmosphere.vert
# End Source File
# Begin Source File

SOURCE=.\GroundFromSpace.frag
# End Source File
# Begin Source File

SOURCE=.\GroundFromSpace.vert
# End Source File
# Begin Source File

SOURCE=.\HDR.vert
# End Source File
# Begin Source File

SOURCE=.\HDRRect.frag
# End Source File
# Begin Source File

SOURCE=.\HDRSquare.frag
# End Source File
# Begin Source File

SOURCE=.\SkyFromAtmosphere.frag
# End Source File
# Begin Source File

SOURCE=.\SkyFromAtmosphere.vert
# End Source File
# Begin Source File

SOURCE=.\SkyFromSpace.frag
# End Source File
# Begin Source File

SOURCE=.\SkyFromSpace.vert
# End Source File
# Begin Source File

SOURCE=.\SpaceFromAtmosphere.frag
# End Source File
# Begin Source File

SOURCE=.\SpaceFromAtmosphere.vert
# End Source File
# Begin Source File

SOURCE=.\SpaceFromSpace.frag
# End Source File
# Begin Source File

SOURCE=.\SpaceFromSpace.vert
# End Source File
# End Group
# Begin Group "Cg Shader Files"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\Common.cg
# End Source File
# Begin Source File

SOURCE=.\GroundFromAtmosphereCg.frag
# End Source File
# Begin Source File

SOURCE=.\GroundFromAtmosphereCg.vert
# End Source File
# Begin Source File

SOURCE=.\GroundFromSpaceCg.frag
# End Source File
# Begin Source File

SOURCE=.\GroundFromSpaceCg.vert
# End Source File
# Begin Source File

SOURCE=.\HDRCg.vert
# End Source File
# Begin Source File

SOURCE=.\HDRRectCg.frag
# End Source File
# Begin Source File

SOURCE=.\HDRSquareCg.frag
# End Source File
# Begin Source File

SOURCE=.\SkyFromAtmosphereCg.frag
# End Source File
# Begin Source File

SOURCE=.\SkyFromAtmosphereCg.vert
# End Source File
# Begin Source File

SOURCE=.\SkyFromSpaceCg.frag
# End Source File
# Begin Source File

SOURCE=.\SkyFromSpaceCg.vert
# End Source File
# Begin Source File

SOURCE=.\SpaceFromAtmosphereCg.frag
# End Source File
# Begin Source File

SOURCE=.\SpaceFromAtmosphereCg.vert
# End Source File
# Begin Source File

SOURCE=.\SpaceFromSpaceCg.frag
# End Source File
# Begin Source File

SOURCE=.\SpaceFromSpaceCg.vert
# End Source File
# End Group
# End Target
# End Project
