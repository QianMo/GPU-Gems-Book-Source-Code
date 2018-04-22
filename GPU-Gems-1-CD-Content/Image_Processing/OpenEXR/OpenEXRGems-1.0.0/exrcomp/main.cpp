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
#include <vector>
#include <cstring>

#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <Iex.h>
#include <clo.h>
#include <Comp.h>

using namespace std;

int
main (int argc, char * argv[])
{
    int status = 0;

    try
    {
	clo::parser parser;
	parser.parse (argc, argv);

	const clo::options & options = parser.get_options ();

	const vector<string> & nonopts = parser.get_non_options ();
	if (nonopts.size () != 4)
	{
	    clo::option_error e ("illegal syntax.");
	    throw e;
	}

	const std::string & exrA = nonopts[0];
	const std::string & op   = nonopts[1];
	const std::string & exrB = nonopts[2];
	const std::string & exrC = nonopts[3];

	if (op != "over" && op != "in" && op != "out")
	{
	    clo::option_error e ("unknown operation");
	    throw e;
	}

	//
	// Open A and B for reading.
	//
	// The images must have the same data and display window,
	// but this requirement should be relaxed.
	//

	Imf::RgbaInputFile inA (exrA.c_str ());
	Imf::RgbaInputFile inB (exrB.c_str ());

	Imath::Box2i dataWinA = inA.dataWindow ();
	Imath::Box2i dataWinB = inB.dataWindow ();

	if ((dataWinA.min.x != dataWinB.min.x) || 
	    (dataWinA.min.y != dataWinB.min.y) ||
	    (dataWinA.max.x != dataWinB.max.x) || 
	    (dataWinA.max.y != dataWinB.max.y))
	{
	    THROW (Iex::BaseExc, "both images must have the same data window");
	}

	Imath::Box2i dpyWinA = inA.displayWindow ();
	Imath::Box2i dpyWinB = inB.displayWindow ();

	if ((dpyWinA.min.x != dpyWinB.min.x) || 
	    (dpyWinA.min.y != dpyWinB.min.y) ||
	    (dpyWinA.max.x != dpyWinB.max.x) || 
	    (dpyWinA.max.y != dpyWinB.max.y))
	{
	    THROW (Iex::BaseExc, "both images must have the same display "
		   << "window");
	}

	//
	// Open C for writing, preserving the data window and display
	// window of the original images.
	//

	Imf::RgbaOutputFile outC (exrC.c_str (), dpyWinA, dataWinA,
				  Imf::WRITE_RGBA);

	//
	// Read A and B.
	//

	Imath::V2i dim (dataWinA.max.x - dataWinA.min.x + 1,
			dataWinA.max.y - dataWinA.min.y + 1);
	int dx = dataWinA.min.x;
	int dy = dataWinA.min.y;

	Imf::Array<Imf::Rgba> imgA (dim.x * dim.y);
	Imf::Array<Imf::Rgba> imgB (dim.x * dim.y);

	inA.setFrameBuffer (imgA - dx - dy * dim.x, 1, dim.x);
	inA.readPixels (dataWinA.min.y, dataWinA.max.y);

	inB.setFrameBuffer (imgB - dx - dy * dim.x, 1, dim.x);
	inB.readPixels (dataWinB.min.y, dataWinB.max.y);

	//
	// Do the comp, overwrite image B with the result.
	//

	if (op == "over")
	    Comp::over (dim, imgA, imgB, imgB);
	else if (op == "in")
	    Comp::in (dim, imgA, imgB, imgB);
	else
	    Comp::out (dim, imgA, imgB, imgB);

	//
	// Write comp'ed image.
	//

	outC.setFrameBuffer (imgB - dx - dy * dim.x, 1, dim.x);
	outC.writePixels (dim.y);
    }
    catch (clo::autoexcept & e)
    {
	switch (e.get_autothrow_id ())
	{
	  case clo::autothrow_help:
	      cout << "Usage: exrcomp [options] <a.exr> <over|in|out> "
		   << "<b.exr> <output.exr>" << endl;
	      cout << e.what ();
	      break;

	  case clo::autothrow_version:
	      cout << "exrcomp version 1.0" << endl;
	      break;

	  default:
	      cerr << "Internal error (illegal autothrow)" << endl;
	      cerr << e.what () << endl;
	      status = 1;
	}
    }
    catch (clo::option_error & e)
    {
	cerr << "exrcomp: " << e.what () << endl;
	cerr << "Use -h for help." << endl;
	status = 1;
    }
    catch (exception & e)
    {
	cerr << "exrcomp: " << e.what () << endl;
	status = 1;
    }
    catch (...)
    {
	cerr << "exrcomp: caught unhandled exception" << endl;
	status = 1;
    }

    return status;
}
