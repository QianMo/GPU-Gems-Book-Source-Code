@echo off

REM make sure we're in the path where the shader is located
REM so that #include directives work properly.
cd %~p1

echo %~nx1

if %~x1==.vsh goto VSH
if %~x1==.vp20 goto VSH
if %~x1==.vp30 goto VSH
if %~x1==.vp40 goto VSH
if %~x1==.gsh goto GSH
if %~x1==.psh goto PSH
if %~x1==.fp20 goto PSH
if %~x1==.fp30 goto PSH
if %~x1==.fp40 goto PSH
echo Sorry, unable to determine shader type for file %~nx1
goto END

:VSH
echo Compiling using vs_4_0...
"C:\Program Files\Microsoft DirectX SDK (December 2006)\Utilities\Bin\x86\fxc.exe" /Tvs_4_0 /Gec %1 
goto END
:GSH
echo Compiling using gs_4_0...
"C:\Program Files\Microsoft DirectX SDK (December 2006)\Utilities\Bin\x86\fxc.exe" /Tgs_4_0 /Gec %1 
goto END
:PSH
echo Compiling using ps_4_0...
"C:\Program Files\Microsoft DirectX SDK (December 2006)\Utilities\Bin\x86\fxc.exe" /Tps_4_0 /Gec %1 
goto END

:END
pause
