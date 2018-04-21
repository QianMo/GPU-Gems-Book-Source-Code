///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2003, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

#include <GlutGlut.h>

#include <ImathVec.h>
#include <GlutWindow.h>

using namespace std;

namespace Glut
{

// static
Glut &
Glut::theGlutContext ()
{
    static Glut theGlut;
    return theGlut;
}


// static
void
Glut::kbHandler (unsigned char key, int x, int y)
{
    int id = glutGetWindow ();
    theGlutContext ()._win[id]->keyboard (key, x, y);
}


// static
void
Glut::idleHandler ()
{
    int id = glutGetWindow ();
    theGlutContext ()._win[id]->idle ();
}


// static
void
Glut::displayHandler ()
{
    int id = glutGetWindow ();
    theGlutContext ()._win[id]->display ();
}


Glut::Glut ()
{
    //
    // Fake args.
    //

    int glutArgc      = 1;
    char * glutArgv[] = {"GlutApp", 0};

    glutInit (&glutArgc, glutArgv);

    glutInitDisplayMode (GLUT_RGBA | GLUT_DOUBLE);
}


Glut::~Glut ()
{
    for (map<int, Window *>::const_iterator i = _win.begin (); i != _win.end (); ++i)
	delete i->second;
}


void
Glut::addWindow (Window * win)
{
    const Imath::V2i & dim = win->dim ();
    glutInitWindowSize (dim.x, dim.y);

    int id = glutCreateWindow (win->name ());

    win->init ();

    //
    // Register the global context's handlers for this window.
    // The global context's handlers route the request to the
    // proper window.
    //

    glutKeyboardFunc (Glut::kbHandler);
    glutIdleFunc (Glut::idleHandler);
    glutDisplayFunc (Glut::displayHandler);

    _win[id] = win;
}


void
Glut::run () const
{
    glutMainLoop ();
}

} // namespace Glut
