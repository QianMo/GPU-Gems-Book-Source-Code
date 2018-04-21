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

#include <iostream>
#include <exception>
#include <vector>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <malloc.h>

#include <ImfRgbaFile.h>
#include <clo.h>
#include <GlutGlut.h>
#include <Display.h>
#include <DisplayLut.h>

using namespace std;

int
main (int argc, char * argv[])
{
    int status = 0;
    vector<Imf::Rgba *> frames;
    ExrPlay::Display * displayWindow = 0;

    try
    {
	clo::parser parser;
	parser.parse (argc, argv);

	const clo::options & options = parser.get_options ();

	const vector<string> & nonopts = parser.get_non_options ();
	if (nonopts.size () < 1)
	{
	    clo::option_error e ("you must specify at least one OpenEXR file.");
	    throw e;
	}

	//
	// Read EXRs.  They must have the same data window.
	//

	Imf::RgbaInputFile in (nonopts[0].c_str ());

	Imath::Box2i win = in.dataWindow ();

	Imath::V2i dim (win.max.x - win.min.x + 1, win.max.y - win.min.y + 1);

	Imf::Rgba * pixels = new Imf::Rgba[dim.x * dim.y];

	frames.push_back (pixels);

	int dx = win.min.x;
	int dy = win.min.y;

	in.setFrameBuffer (frames[0] - dx - dy * dim.x, 1, dim.x);
	in.readPixels (win.min.y, win.max.y);

	for (int i = 1; i != nonopts.size (); ++i)
	{
	    Imf::RgbaInputFile nextIn (nonopts[i].c_str ());

	    Imath::Box2i nextWin = nextIn.dataWindow ();

	    if ((nextWin.min.x != win.min.x) || (nextWin.min.y != win.min.y) ||
		(nextWin.max.x != win.max.x) || (nextWin.max.y != win.max.y))
	    {
		THROW (Iex::BaseExc, "images must have the same data window ("
		       << nonopts[i] << " differs from " << nonopts[0] << ")");
	    }

	    pixels = new Imf::Rgba[dim.x * dim.y];

	    frames.push_back (pixels);

	    nextIn.setFrameBuffer (frames[i] - dx - dy * dim.x, 1, dim.x);
	    nextIn.readPixels (win.min.y, win.max.y);
	}
	
	Glut::Glut & glut = Glut::Glut::theGlutContext ();

	if (!options.lut.empty ())
	    displayWindow = new ExrPlay::DisplayLut (frames,
						     dim,
						     options.rate,
						     options.lut,
						     options.builtin,
						     options.usePdr);
	else
	    displayWindow = new ExrPlay::Display (frames,
						  dim, 
						  options.rate,
						  options.builtin,
						  options.usePdr);

	glut.addWindow (displayWindow);
	glut.run ();
    }
    catch (clo::autoexcept & e)
    {
	switch (e.get_autothrow_id ())
	{
	  case clo::autothrow_help:
	      cout << "Usage: exrplay [options] <exrfile1> [exrfile2] ..." << endl;
	      cout << e.what ();
	      break;

	  case clo::autothrow_version:
	      cout << "exrplay version 1.0" << endl;
	      break;

	  default:
	      cerr << "Internal error (illegal autothrow)" << endl;
	      cerr << e.what () << endl;
	      status = 1;
	}
    }
    catch (clo::option_error & e)
    {
	cerr << "exrplay: " << e.what () << endl;
	cerr << "Use -h for help." << endl;
	status = 1;
    }
    catch (exception & e)
    {
	cerr << "exrplay: " << e.what () << endl;
	status = 1;
    }
    catch (...)
    {
	cerr << "exrplay: caught unhandled exception" << endl;
	status = 1;
    }

    delete displayWindow;
    for (int i = 0; i != frames.size (); ++i)
	delete [] frames[i];

    return status;
}
