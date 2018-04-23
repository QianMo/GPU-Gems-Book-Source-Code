/////////////////////////////////////////////////////////////////////////////
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
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
// (This is the Modified BSD License)
/////////////////////////////////////////////////////////////////////////////


#ifndef GRIDFILE_H
#define GRIDFILE_H

/*
  Provides basic support for reading in grid files dumped out by Gelato.
  Basic routines for reading in the files and accessing the list of grids
  and all the values stored on a grid are provided.
  
  You can dump a grid file from Gelato with:
  
      Attribute ("string griddump:filename", "filename.grid")

  Here's an example of how to use this library to read a gridfile:

      #include "gridfile.h"

      GridFile gf;
      if (!gf.read ("filename.grid")) {
          fprintf (stderr, "%s\n", gf.errorstr());
          exit (-1);
      }
      for (size_t i = 0; i < gf.gridlist.size(); i++) {
          Grid *g = gf.gridlist[i];
          int nu, nv;
          Vector3 *P, *C;
          if (!g->data ("P", nu, nv, &P)) {
              fprintf (stderr, "Cannot find P in grid\n");
              exit (-1);
          }
          if (!g->data ("C", nu, nv, &C)) {
              fprintf (stderr, "Cannot find C in grid\n");
              exit (-1);
          }

          < render the grid with position and color >
      }

      
  Currently, this library only supports floating point scalar and
  3-tuple grid data.  It should be easy to extend to other types.
  
*/


#include <vector>

#include "paramtype.h"
#include "vecmat.h"


namespace Gelato {

    
class Grid {
 public:
    enum GridType { QUADMESH, CURVES };
    
    Grid (GridType gridtype) : gridtype (gridtype), motion (false) {}
    ~Grid ();

    // returns the type of this grid
    GridType type () const { return gridtype; }
            
    // returns true if data found and sets ns, nt, and data
    bool data (const char *name, int &ns, int &nt, Vector3 **data) const;
    bool data (const char *name, int &ns, int &nt, float **data) const;

    // add a new data member, data pointer is stored, so don't free it
    void add_data (const char *name, ParamBaseType type, int ns, int nt,
        void *data) {
        GridData *gdata = new GridData (name, type, ns, nt, data);
        datalist.push_back (gdata);
    }
    
    // If motion blurred, return true and put the begin and end times in
    // times[0] and [1].  If not motion blurred, return false.
    bool get_times (float *times) {
        if (! motion)
            return false;
        times[0] = time0;
        times[1] = time1;
        return true;
    }
 private:
    
    struct GridData {
        GridData (const char *name, ParamBaseType type, int ns, int nt,
            void *data) : name (strdup(name)), type (type), ns (ns), nt (nt),
                data(data) {}
        ~GridData () { free (name); free (data); }

        char *name;
        ParamBaseType type;
        int ns, nt;
        void *data;
    };

    GridType gridtype;
    std::vector <GridData *> datalist;
    float time0, time1;
    bool motion;
    friend class GridFile;
};



class GridFile {
 public:
    GridFile () : errstr (NULL) {}
    ~GridFile ();

    // Optional grid callback returns true to keep the grid, false to cull
    bool read (const char *filename, bool (*gridcb)(Grid *, void *) = NULL,
        void *user_data = NULL);
    const char *errorstr () const { return errstr; }

    std::vector <Grid *> gridlist;
    
 private:
    char *errstr;

    bool read_grid (FILE *fd);
};

 
};      // namespace Gelato


#endif // GRIDFILE_H
