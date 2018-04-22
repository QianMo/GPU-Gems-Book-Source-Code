# Microsoft Developer Studio Project File - Name="cinema4dsdk" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** NICHT BEARBEITEN **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

CFG=cinema4dsdk - Win32 Release
!MESSAGE Dies ist kein gültiges Makefile. Zum Erstellen dieses Projekts mit NMAKE
!MESSAGE verwenden Sie den Befehl "Makefile exportieren" und führen Sie den Befehl
!MESSAGE 
!MESSAGE NMAKE /f "CINEMA4DSDK.mak".
!MESSAGE 
!MESSAGE Sie können beim Ausführen von NMAKE eine Konfiguration angeben
!MESSAGE durch Definieren des Makros CFG in der Befehlszeile. Zum Beispiel:
!MESSAGE 
!MESSAGE NMAKE /f "CINEMA4DSDK.mak" CFG="cinema4dsdk - Win32 Release"
!MESSAGE 
!MESSAGE Für die Konfiguration stehen zur Auswahl:
!MESSAGE 
!MESSAGE "cinema4dsdk - Win32 Release" (basierend auf  "Win32 (x86) Dynamic-Link Library")
!MESSAGE "cinema4dsdk - Win32 Debug" (basierend auf  "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "cinema4dsdk - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "./obj/sdk_rel"
# PROP Intermediate_Dir "./obj/sdk_rel"
# PROP Ignore_Export_Lib 1
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "PLUGIN_EXPORTS" /Yu"stdafx.h" /FD /c
# ADD CPP /nologo /G6 /MT /W3 /vmg /vms /GX- /O2 /I ".\res" /I ".\res\description" /I "..\..\resource\_api" /I "..\..\resource\_api\c4d_customgui" /I "..\..\resource\_api\c4d_libs" /I "..\..\resource\_api\c4d_preview" /I "..\..\resource\_api\c4d_scaling" /I "..\..\resource\_api\c4d_gv" /I "$(CG_INC_PATH)" /D "WIN32" /D "__PC" /D "NDEBUG" /D "_WINDOWS" /YX /FD /c
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib glu32.lib opengl32.lib CgFXParser.lib winmm.lib /nologo /dll /machine:I386 /nodefaultlib:"LIBCD" /out:"c4dfx.cdl" /libpath:"$(CG_LIB_PATH)"

!ELSEIF  "$(CFG)" == "cinema4dsdk - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "./obj/sdk_deb"
# PROP Intermediate_Dir "./obj/sdk_deb"
# PROP Ignore_Export_Lib 1
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "PLUGIN_EXPORTS" /Yu"stdafx.h" /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /vmg /vms /GX- /ZI /Od /I ".\res" /I ".\res\description" /I "..\..\resource\_api" /I "..\..\resource\_api\c4d_customgui" /I "..\..\resource\_api\c4d_libs" /I "..\..\resource\_api\c4d_preview" /I "..\..\resource\_api\c4d_scaling" /I "..\..\resource\_api\c4d_gv" /I "$(CG_INC_PATH)" /D "WIN32" /D "__PC" /D "_DEBUG" /D "_WINDOWS" /YX /FD /GZ /c
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib glu32.lib opengl32.lib CgFXParser.lib winmm.lib /nologo /dll /debug /machine:I386 /nodefaultlib:"LIBCD" /out:"c4dfx.cdl" /pdbtype:sept /libpath:"$(CG_LIB_PATH)"

!ENDIF 

# Begin Target

# Name "cinema4dsdk - Win32 Release"
# Name "cinema4dsdk - Win32 Debug"
# Begin Group "Source Code"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\source\BitmapWrapper.cpp
# End Source File
# Begin Source File

SOURCE=.\source\BitmapWrapper.h
# End Source File
# Begin Source File

SOURCE=.\source\C4DWrapper.cpp
# End Source File
# Begin Source File

SOURCE=.\source\C4DWrapper.h
# End Source File
# Begin Source File

SOURCE=.\source\FXCommand.cpp
# End Source File
# Begin Source File

SOURCE=.\source\FXDialog.cpp
# End Source File
# Begin Source File

SOURCE=.\source\FXDialog.h
# End Source File
# Begin Source File

SOURCE=.\source\FXMaterial.cpp
# End Source File
# Begin Source File

SOURCE=.\source\FXMaterial.h
# End Source File
# Begin Source File

SOURCE=.\source\FXWrapper.cpp
# End Source File
# Begin Source File

SOURCE=.\source\FXWrapper.h
# End Source File
# Begin Source File

SOURCE=.\source\IDnumbers.h
# End Source File
# Begin Source File

SOURCE=.\source\LightsMaterialsObjects.cpp
# End Source File
# Begin Source File

SOURCE=.\source\LightsMaterialsObjects.h
# End Source File
# Begin Source File

SOURCE=.\source\LightTag.cpp
# End Source File
# Begin Source File

SOURCE=.\source\Main.cpp
# End Source File
# Begin Source File

SOURCE=.\source\MatrixMath.cpp
# End Source File
# Begin Source File

SOURCE=.\source\MatrixMath.h
# End Source File
# Begin Source File

SOURCE=.\source\nv_dds.cpp
# End Source File
# Begin Source File

SOURCE=.\source\nv_dds.h
# End Source File
# Begin Source File

SOURCE=.\source\Paint.cpp
# End Source File
# Begin Source File

SOURCE=.\source\Paint.h
# End Source File
# Begin Source File

SOURCE=.\source\PrefDialog.cpp
# End Source File
# Begin Source File

SOURCE=.\source\PrefDialog.h
# End Source File
# Begin Source File

SOURCE=.\source\Render.cpp
# End Source File
# Begin Source File

SOURCE=.\source\Render.h
# End Source File
# Begin Source File

SOURCE=.\source\RenderDialog.cpp
# End Source File
# Begin Source File

SOURCE=.\source\RenderDialog.h
# End Source File
# Begin Source File

SOURCE=.\source\WindowAdapter.cpp
# End Source File
# Begin Source File

SOURCE=.\source\WindowAdapter.h
# End Source File
# Begin Source File

SOURCE=.\source\WinHack.h
# End Source File
# Begin Source File

SOURCE=.\source\WinInit.cpp
# End Source File
# Begin Source File

SOURCE=.\source\WinInit.h
# End Source File
# End Group
# End Target
# End Project
