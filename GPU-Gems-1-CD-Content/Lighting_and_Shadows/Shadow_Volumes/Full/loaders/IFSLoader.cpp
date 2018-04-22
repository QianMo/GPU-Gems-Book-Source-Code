/**
 @maintainer Morgan McGuire, matrix@graphics3d.com
 @cite       Written by Nate Miller, nathanm@uci.edu
 @created    2002-08-10
 @edited     2002-08-10
 */

#include "IFSLoader.h"

IFSReader::IFSReader() {
   verts = 0;
   tris = 0;
   numVerts = 0;
   numTris = 0;
}


IFSReader::~IFSReader() {
   delete [] verts;
   delete [] tris;
}


void IFSReader::load(
    const std::string&  name,
    IFSModel&           dest) {

   delete [] verts;
   delete [] tris;

   verts = 0;
   tris = 0;
   numVerts = 0;
   numTris = 0;

   BinaryInput reader(name, BinaryInput::LITTLE_ENDIAN);

   if (!reader.getLength())
      throw std::string("Failed to open " + name);

   processHeader(reader);

   while (reader.hasMore()) {
      std::string str = reader.readString32();

      if (str == "VERTICES") {
         debugAssertM(!verts, "Multiple vertex fields!");
         processVerts(reader);
      } else if (str == "TRIANGLES") {
         debugAssertM(!tris, "Multiple triangle fields!");
         processTris(reader);
      }
   }

   dest.setup(meshName, numVerts, verts, numTris, tris);

   verts = 0;
   tris = 0;
}


void IFSReader::processHeader(BinaryInput& input) {
   std::string str = input.readString32();

   if (str != "IFS") {
      throw std::string("File is not an IFS file");
   }

   if (input.readFloat32() != 1.0f) {
      throw std::string("Bad IFS version, expecting 1.0");
   }
   
   meshName = input.readString32();
}


void IFSReader::processVerts(BinaryInput& input) {
   uint32 num = input.readUInt32();

   if (!num) {
      throw std::string("Bad number of vertices");
   }

   numVerts = num;
   verts = new IFSVertex[num];

   for (int i = 0; i < num; ++i) {
      verts[i].setPosition(input.readVector3());
   }
}


void IFSReader::processTris(BinaryInput& input) {
   uint32 num = input.readUInt32();

   if (!num) {
      throw std::string("Bad number of triangles");
   }

   numTris = num;
   tris = new IFSTriangle[num];

   for (int i = 0; i < num; ++i) {
      uint32 a = input.readUInt32();
      uint32 b = input.readUInt32();
      uint32 c = input.readUInt32();

      tris[i].setIndices(a, b, c);
   }
}