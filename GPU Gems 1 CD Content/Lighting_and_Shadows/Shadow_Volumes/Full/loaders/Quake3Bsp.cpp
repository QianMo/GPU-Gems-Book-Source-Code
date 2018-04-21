/**
  @file Quake3Bsp.cpp

  @maintainer Kevin Egan (ktegan@cs.brown.edu)
  @cite BenHumphrey (digiben@gametutorials.com)
  @cite Kekoa Proudfoot (kekoa@graphics.stanford.edu)

*/

#include "../BasicModel.h"
#include "../Renderer.h"
#include "Quake3Bsp.h"

struct Q3BspHeader
{
    char                        ID[4];
    int                         version;
};

struct Q3BspLump
{
    int                         offset;
    int                         length;

    enum Lumps
    {
        Entities = 0,
        Textures,
        Planes,
        Nodes,
        Leafs,
        LeafFaces,
        LeafBrushes,
        Models,
        Brushes,
        BrushSides,
        Vertices,
        Indices,
        Shaders,
        Faces,
        Lightmaps,
        LightVolumes,
        VisData,
        NumLumps
    };
};


struct Q3BspVec2
{
    float                       x;
    float                       y;
};


struct Q3BspVec3
{
    float                       x;
    float                       y;
    float                       z;
};


struct Q3BspVertex
{
    Q3BspVec3                   position;
    Q3BspVec2                   texCoord;
    Q3BspVec2                   lightmapCoord;
    Q3BspVec3                   normal;
    unsigned char               color[4];
};


struct Q3BspFace
{
    int                         texID;
    int                         effect;
    int                         faceType;
    int                         startVertexIndex;
    int                         numVertices;
    int                         indexOffset;
    int                         numIndices;
    int                         lightmapID;
    int                         lightmapCorner[2];
    int                         lightmapSize[2];
    Q3BspVec3                   lightmapPos;
    Q3BspVec3                   lightmapBasis[2];
    Q3BspVec3                   normal;
    int                         patchSize[2];

    enum FaceTypes {
        FACE_POLYGON = 1,
        FACE_PATCH = 2,
        FACE_MESH = 3,
        FACE_BILLBOARD = 4
    };
};


struct Q3BspTexture
{
    char                        filename[64];
    int                         flags;
    int                         contents;
};


/**
 * This creates a BasicModel object using the static Quake 3
 * level geometry.
 */
BasicModel* loadQuake3Bsp(
        const std::string&          filename,
        int                         tesselationLevel,
        float                       scale)
{
    FILE*       fp;
    int         ret;
    int         i;
    int         j;
    int         k;

    fp = fopen(filename.c_str(), "rb");
    if (fp == NULL) {
        debugAssertM(false, std::string("could not open the Q3Bsp file ") +
                filename);
        return NULL;
    }

    Q3BspHeader header;
    Q3BspLump lumpsArray[Q3BspLump::NumLumps];

    // read in header information
    ret = fread(& header, 1, sizeof(header), fp);
    ret = fread(& lumpsArray, Q3BspLump::NumLumps, sizeof(lumpsArray[0]), fp);
    if ((strncmp(header.ID, "IBSP", 4) != 0) || (header.version != 0x2e)) {
        debugAssertM(false, std::string("Q3Bsp file not the valid format ") +
                filename);
        return NULL;
    }

    // read in lump information
    int numVertices = lumpsArray[Q3BspLump::Vertices].length /
        sizeof(Q3BspVertex);
    Q3BspVertex* vertexData = new Q3BspVertex[numVertices];

    int numIndices = lumpsArray[Q3BspLump::Indices].length /
        sizeof(int);
    int* indexData = new int[numIndices];

    int numFaces = lumpsArray[Q3BspLump::Faces].length /
        sizeof(Q3BspFace);
    Q3BspFace* faceData = new Q3BspFace[numFaces];

    int numTextures = lumpsArray[Q3BspLump::Textures].length /
        sizeof(Q3BspTexture);
    Q3BspTexture* textureData = new Q3BspTexture[numTextures];

    // read in vertex, face and texture data
    fseek(fp, lumpsArray[Q3BspLump::Vertices].offset, SEEK_SET);
    fread(vertexData, numVertices, sizeof(Q3BspVertex), fp);

    fseek(fp, lumpsArray[Q3BspLump::Indices].offset, SEEK_SET);
    fread(indexData, numIndices, sizeof(int), fp);

    fseek(fp, lumpsArray[Q3BspLump::Faces].offset, SEEK_SET);
    fread(faceData, numFaces, sizeof(Q3BspFace), fp);

    fseek(fp, lumpsArray[Q3BspLump::Textures].offset, SEEK_SET);
    fread(textureData, numTextures, sizeof(Q3BspTexture), fp);

    fclose(fp);

    Array<Vector3> vertexArray(numVertices);
    Array<Vector2> texCoordArray(0);
    Array<BasicFace> faceArray(0);
    Array<BasicEdge> edgeArray(0);
    Color3 color(1, 1, 1);


    // XXX if you can find a level that has no t-junctions or other
    // bad properties then you can try setting this to true
    bool castShadow = false;


    // process vertices
    for (i = 0; i < numVertices; i++) {
        vertexArray[i][0] = vertexData[i].position.x * scale;
        vertexArray[i][1] = vertexData[i].position.z * scale;
        vertexArray[i][2] = -vertexData[i].position.y * scale;

        // XXX if you wanted texture coordinates set that here
        // untested code below
        //texCoordData[i].x = vertexData[i].texCoord.x;
        //texCoordData[i].y = -vertexData[i].texCoord.y;
    }

    // process faces
    for (i = 0; i < numFaces; i++) {
        switch (faceData[i].faceType) {
        case Q3BspFace::FACE_POLYGON :
            for (j = 0; j < faceData[i].numVertices - 2; j++) {
                BasicFace newFace;
                newFace.vertexIndices[0] = faceData[i].startVertexIndex;
                newFace.vertexIndices[1] = faceData[i].startVertexIndex + j + 2;
                newFace.vertexIndices[2] = faceData[i].startVertexIndex + j + 1;
                faceArray.push(newFace);
            }
            break;

        case Q3BspFace::FACE_PATCH :

            // XXX not totally sure these faces are added in the correct
            // order.  Also this currently does not tesselate any faces
            // besides the control points of the patch.
            debugAssert(faceData[i].numVertices ==
                    faceData[i].patchSize[0] * faceData[i].patchSize[1]);

            for (j = 0; j < faceData[i].patchSize[1] - 1; j++) {
                for (k = 0; k < faceData[i].patchSize[0] - 1; k++) {
                    #define face(row, col) \
                        (faceData[i].startVertexIndex + \
                        (row) * faceData[i].patchSize[0] + (col))

                    BasicFace newFace;
                    newFace.vertexIndices[0] = face(j    ,  k    );
                    newFace.vertexIndices[1] = face(j + 1,  k    );
                    newFace.vertexIndices[2] = face(j    ,  k + 1);
                    faceArray.push(newFace);

                    newFace.vertexIndices[0] = face(j + 1,  k    );
                    newFace.vertexIndices[1] = face(j + 1,  k + 1);
                    newFace.vertexIndices[2] = face(j    ,  k + 1);
                    faceArray.push(newFace);

                    #undef face
                }
            }
            break;

        case Q3BspFace::FACE_MESH :
            debugAssert(faceData[i].numIndices % 3 == 0);
            for (j = 0; j < faceData[i].numIndices; j += 3) {
                BasicFace newFace;
                newFace.vertexIndices[0] = faceData[i].startVertexIndex +
                    indexData[faceData[i].indexOffset + j + 0];
                newFace.vertexIndices[1] = faceData[i].startVertexIndex +
                    indexData[faceData[i].indexOffset + j + 2];
                newFace.vertexIndices[2] = faceData[i].startVertexIndex +
                    indexData[faceData[i].indexOffset + j + 1];
                faceArray.push(newFace);
                debugAssert(faceData[i].indexOffset >= 0);
                debugAssert(newFace.vertexIndices[0] >= 0);
                debugAssert(newFace.vertexIndices[1] >= 0);
                debugAssert(newFace.vertexIndices[2] >= 0);
            }
            break;
        }
    }

    // process textures
    for (i = 0; i < numTextures; i++) {
        // XXX if you want textures you need to do some processing here
    }

    delete[] vertexData;
    delete[] indexData;
    delete[] faceData;
    delete[] textureData;

    BasicModel* finalModel = new BasicModel(vertexArray, texCoordArray,
        faceArray, edgeArray, castShadow, color);

    if (tesselationLevel > 0) {
        finalModel->compact();
        finalModel->retesselateFaces(tesselationLevel);
        finalModel->compact();
    }

    if (castShadow) {
        finalModel->computeEdges();
    }

    return finalModel;
}


