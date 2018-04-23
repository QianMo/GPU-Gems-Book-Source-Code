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

#include <iostream>
#include <fstream>
#include <errno.h>
#include "dassert.h"
#include "gridfile.h"
#include "gelendian.h"


namespace Gelato {


Grid::~Grid ()
{
    for (size_t i = 0; i < datalist.size(); i++)
        delete datalist[i];
}



bool
Grid::data (const char *name, int &ns, int &nt, Vector3 **data) const
{
    *data = NULL;
    
    for (size_t i = 0; i < datalist.size(); i++) {
        // find an entry that matches the name and is a Vector3
        if (strcmp (datalist[i]->name, name) == 0 &&
            (datalist[i]->type == PT_POINT ||
                datalist[i]->type == PT_VECTOR ||
                datalist[i]->type == PT_NORMAL ||
                datalist[i]->type == PT_COLOR)) {
            ns = datalist[i]->ns;
            nt = datalist[i]->nt;
            *data = (Vector3 *)datalist[i]->data;
            return true;
        }
    }

    return false;
}


bool
Grid::data (const char * name, int &ns, int &nt, float **data) const
{
    *data = NULL;
    
    for (size_t i = 0; i < datalist.size(); i++) {
        // find an entry that matches the name and is a Vector3
        if (strcmp (datalist[i]->name, name) == 0 &&
            datalist[i]->type == PT_FLOAT) {
            ns = datalist[i]->ns;
            nt = datalist[i]->nt;
            *data = (float *)datalist[i]->data;
            return true;
        }
    }

    return false;
}



GridFile::~GridFile ()
{
    for (size_t i = 0; i < gridlist.size(); i++)
        delete gridlist[i];
}



// Read a line into linebuf, realloc if it's not big enough.
static void
getline (FILE *fd, char* &linebuf, int &linesize)
{
    // If linebuf is not yet allocated, give it 1k
    if (! linebuf) {
        linesize = 1024;
        linebuf = (char *) malloc (linesize);
    }
    int len = 0;
    static int hostendian = Gelato::bigendian();  // 1 if big
    while (!feof(fd)) {
        unsigned char c = fgetc (fd);
        if (c >= 0200 && c <= 0203) {
            bool swap = ((c & 1) != hostendian);
            linebuf[len++] = c;
            DASSERT (sizeof(int) == sizeof(float));
            int arraylen;
            fread (&arraylen, sizeof(int), 1, fd);
            if (swap)
                swap_endian (&arraylen);
            // Make space for the array length as well as the array
            int toadd = (arraylen+1)*sizeof(float) + 1;
            if (len+toadd >= linesize-1) {
                linesize += std::max (linesize, 2*toadd);
                linebuf = (char *) realloc (linebuf, linesize);
            }
            // put arraylen into linebuf
            for (int i = 0;  i < 4;  ++i)
                linebuf[len++] = ((char *)&arraylen)[i];
            fread (linebuf+len, sizeof(float), arraylen, fd);
            if (swap)
                swap_endian (linebuf+len, arraylen);
            len += arraylen*sizeof(float);
            continue;
        }
        if (c == '\n') {    // End of line?  we're done
            linebuf[len] = 0;
            return;
        }
        linebuf[len++] = c;
        // Make sure there's space for the next char.  Double the size
        // if there's no more room.
        if (len >= linesize-1) {
            linesize *= 2;
            linebuf = (char *) realloc (linebuf, linesize);
        }
    }
    linebuf[len] = 0;
}

// Advance p while *p is a whitespace char
inline void skipwhite (char* &p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
        ++p;
}

// Advance p until you point to one of the stop chars (or end of string)
inline void skipuntil (char* &p, char stop1, char stop2=0) {
    while (*p && *p != stop1 && *p != stop2) ++p;
}

// Advance p until you see one of the stop chars, and eat the stop char.
inline void skipincluding (char* &p, char stop1, char stop2=0) {
    while (*p && *p != stop1 && *p != stop2) ++p;
    if (*p && (*p == stop1 || *p == stop2)) ++p;
}


static void read_array (char* &p, int n, int nfloats, float *data)
{
    skipwhite (p);
    if ((unsigned char)*p == 0200 || (unsigned char)*p == 0201) {
        ASSERT(0);  // no idea what to do with int array
    }
    if ((unsigned char)*p == 0202 || (unsigned char)*p == 0203) {
        // getline already has converted a binary array to the
        // endianness of this host
        ++p;
        int len;
        memcpy (&len, p, sizeof(int));
        ASSERT (len == n*nfloats);
        p += sizeof(int);
        memcpy (data, p, len*sizeof(float));
        p += len*sizeof(float);
        skipincluding (p, ',', ')');
    } else {
        // Read the list of floats
        skipincluding (p, '(');
        for (int i = 0;  i < n*nfloats;  ++i) {
            while (!isdigit (*p) && *p != '-' && *p != '+') ++p;
            data[i] = (float)atof (p);
            skipincluding (p, ',', ')');
        }
    }
}



// Parse the given file of dumped grids (which are Patch calls only,
// possibly with Motion followed by two Patch calls).
bool
GridFile::read (const char *filename, bool (*gridcb)(Grid *, void *),
                void *user_data)
{
    // Try to open the file
    FILE *fd = fopen (filename, "rb");
    if (fd == NULL) {
        errstr = strerror (errno);
        return false;
    }

    char *linebuf = NULL;
    int linesize = 0;
    char buf[4096];
    char *err;
    int lineno = 0;
    Grid *grid = NULL;
    float *data = NULL;
    char interptoken[1024], typetoken[1024], nametoken[1024];
    int ns, nt;
    bool inmotion = false;
    int motiontime = 0;
    float time0, time1;

    while (! feof(fd)) {
        // Read a line
        ++lineno;
        getline (fd, linebuf, linesize);
        char *p = linebuf;
        skipwhite (p);

        if (! strncmp (p, "Patch", 5)) {
            // This is a Patch call.  If we're not motion blurred, or we
            // are the first time, create a new Grid and add it to the
            // grid list.  Subsequent times will use the same grid.
            if (motiontime == 0) {
                grid = new Grid (Grid::QUADMESH); // a new grid to store data
                gridlist.push_back (grid);        // store new grid pointer
                if (inmotion) {
                    grid->time0 = time0;
                    grid->time1 = time1;
                    grid->motion = true;
                }
            }

            // Skip up to the parameters.
            p += 5;
            skipincluding (p, '\"');
            skipincluding (p, '\"');
            skipincluding (p, ',');
            skipwhite (p);
            ns = atoi (p);
            skipincluding (p, ',');
            nt = atoi (p);
            skipincluding (p, ',');
            
            while (*p) {
                // Read the quoted string with the param type and name
                skipincluding (p, '\"');
                if (! *p)
                    break;
                char *typestring = p;
                skipuntil (p, '\"');
                *p++ = 0;
                sscanf (typestring, "%s %s %s",
                        interptoken, typetoken, nametoken);
                int n = 0;
                if (! strcmp (interptoken, "vertex"))
                    n = ns * nt;
                else if (! strcmp (interptoken, "linear"))
                    n = 4;
                else n = 1;
                ParamType ptype (typetoken);
                ParamBaseType basetype = (ParamBaseType)ptype.basetype;
                int nfloats = ParamBaseTypeNFloats (basetype);
                if (nfloats > 0) {
                    data = (float *)malloc (n*nfloats * sizeof(float));
                }

                // FIXME - should handle strings, too

                skipwhite (p);
                if (*p == ',')
                    ++p;

                read_array (p, n, nfloats, data);
                
                if (inmotion && motiontime >= 1) {
                    // For the second Patch in a Motion block, pass along
                    // "P" as "$P1", and just forget about all other data.
                    if (! strcmp (nametoken, "P")) {
                        strcpy (nametoken, "$P1");
                    } else {
                        free (data);
                        data = NULL;
                    }
                }
                if (data) {
                    grid->add_data (nametoken, basetype, ns, nt, data);
                }
            }

            // User callback, if specified
            if (inmotion == false || motiontime >= 2)
                if (gridcb)
                    (*gridcb) (grid, user_data);

            // Increment the motion counter.  If we're done with the
            // motion block, reset the state machine.
            if (inmotion && ++motiontime >= 2) {
                inmotion = false;
                motiontime = 0;
            }

        } else if (! strncmp (p, "Motion", 6)) {
            // If we hit a 'Motion' call, grab the two times and set the
            // flags indicating that we expect two Patch calls next.
            skipincluding (p, '(');
            sscanf (p, "%f, %f", &time0, &time1);
            inmotion = true;
            motiontime = 0;

        } else if (! strncmp (p, "Curves", 6)) {
            if (motiontime == 0) {
                grid = new Grid (Grid::CURVES);   // new grid to store data
                gridlist.push_back (grid);        // store new grid pointer
                if (inmotion) {
                    grid->time0 = time0;
                    grid->time1 = time1;
                    grid->motion = true;
                }
            }
            // Skip up to the parameters.
            p += 6;
            skipincluding (p, '\"');   // skip over interp type?
            skipincluding (p, '\"');
            skipincluding (p, ',');
            skipwhite (p);
            ns = atoi (p);             // # curves
            skipincluding (p, ',');
            nt = atoi (p);             // # verts / curve
            skipincluding (p, ',');
            
            while (*p) {
                skipincluding (p, '\"');
                if (! *p)
                    break;
                char *typestring = p;
                skipuntil (p, '\"');
                *p++ = 0;
                sscanf (typestring, "%s %s %s",
                        interptoken, typetoken, nametoken);
                int n = 0;
                if (strcmp (interptoken, "vertex")) {
                    err = "Cannot parse anything but \"vertex\" curve "
                        "data interpolation types.\n";
                    goto error;
                }
                n = ns * nt;
                ParamType ptype (typetoken);
                ParamBaseType basetype = (ParamBaseType)ptype.basetype;
                int nfloats = ParamBaseTypeNFloats (basetype);
                if (nfloats > 0) {
                    data = (float *)malloc (n*nfloats * sizeof(float));
                }

                // FIXME - should handle strings, too

                skipwhite (p);
                if (*p == ',')
                    ++p;

                read_array (p, n, nfloats, data);

                if (inmotion && motiontime >= 1) {
                    // For the second Patch in a Motion block, pass along
                    // "P" as "$P1", and just forget about all other data.
                    if (! strcmp (nametoken, "P")) {
                        strcpy (nametoken, "$P1");
                    } else {
                        free (data);
                        data = NULL;
                    }
                }
                if (data) {
                    grid->add_data (nametoken, basetype, ns, nt, data);
                }
            }

            // User callback, if specified
            if (inmotion == false || motiontime >= 2)
                if (gridcb)
                    (*gridcb) (grid, user_data);

            // Increment the motion counter.  If we're done with the
            // motion block, reset the state machine.
            if (inmotion && ++motiontime >= 2) {
                inmotion = false;
                motiontime = 0;
            }

        } else {
            // Unknown -- skip
        }
    }

    return true;
    
 error:
    sprintf (buf, "ERROR: %s at line %d in \"%s\"", err, lineno, filename);
    errstr = strdup (buf);
    fclose (fd);
    return false;
}


};    // namespace Gelato

