/**
 @maintainer Morgan McGuire, matrix@graphics3d.com
 @cite       Written by Nate Miller, nathanm@uci.edu
 @created    2002-08-10
 @edited     2002-08-10
 */

#ifndef IFSREADER_H
#define IFSREADER_H

#include "IFSModel.h"
#include <g3dAll.h>

class IFSReader {
private:
   std::string          meshName;
   IFSVertex*           verts;
   IFSTriangle*         tris;
   uint32               numVerts;
   uint32               numTris;

   void processHeader(BinaryInput& input);
   void processVerts(BinaryInput& input);
   void processTris(BinaryInput& input);

public:
   IFSReader();
   virtual ~IFSReader();

   /**
    Load the model with the passed name and place it into dest.
    throws std::string if the file is corrupt.
    */
   void load(const std::string& name, IFSModel& dest);
};

#endif