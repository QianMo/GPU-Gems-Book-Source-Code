//***********************************************************************
//											 
//		- "Talk to me like I'm a 3 year old!" Programming Lessons -		 
//                                                                       
//		$Author:	DigiBen		digiben@gametutorials.com	 
//											
//		$Program:	SkyBox
//											
//		$Description:	This shows how to create a textured sky box		 
//											
//		$Date:		11/2/01						
//											
//***********************************************************************


Files:  	Main.cpp   (The Source File containing the most worked with code)
		Init.cpp   (The Source File containing the rarely changed code)
		Camera.cpp (The Source File containing the camera code)
		Camera.h   (The header File containing the camera class and prototypes)
		Main.h     (The header file that holds the global variables and prototypes)
		SkyBox.dsp  (The Project File holding the project info)
		SkyBox.dsw  (The Workspace File holding the workspace info)

Controls Used:	w, s, UP_ARROW, DOWN_ARROW - Move the camera forward and backward
		a, d, RIGHT_ARROW, LEFT_ARROW - Strafe the camera left and right
		Mouse movement - Moves the view for first person mode
		Left Mouse Button - Turns wireframe on and off

Libraries:      opengl32.lib glu32.lib gluax.lib winmm.lib

Win32 App:	Remember, if using visual c/c++, if you want to create a windows
		application from scratch, and not a console application, you must 
		choose "Win32 App" in the projects tab when selecting "New" from
		the file menu in visual C.  Your program will not be able to run
		if you try and create a window inside of a console program.  Once
		again, if creating a new project, click on "Win32 Application" and
		not "Win32 Console Application" if you want to create a program for
		windows (not a DOS window).  This process is further explained at 
		www.GameTutorials.com

Instructions:	If you have visual studio c++ (around version 6) just click on the
		<Program Name>.dsw file.  This will open up up visual c++.  You will most
		likely see the code for <Program Name>.cpp.  If you don't, there should be
		a tab on the left window (workspace window) called "FileView".  Click the 
		tab and click on the plus sign in this tab if it isn't already open.
		There you should see 2 folders called "source" and "header".
		Double click on the "source" folder and you should see your source file with
		a .cpp extension after it.  Double click on the file and it will open
		up in your main window.  Hit Control-F5 to run the program.
		You will probably see a prompt to compile/build the project.  Click OK and
		a window should pop up with the program. :)

Ben Humphrey
Game Programmer
digiben@gametutorials.com
www.GameTutorials.com