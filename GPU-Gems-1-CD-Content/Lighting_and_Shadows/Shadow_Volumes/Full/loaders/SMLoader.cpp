/**
  @file SMLoader.cpp

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)

*/


#include <iostream>
#include <fstream>
#include <sstream>
#include "../BasicModel.h"
#include "SMLoader.h"

using namespace std;

BasicModel* importSMFile(
        const std::string&      filePath)
{
    enum ImportState {
        VERTICES_HEADER = 0,
        VERTICES_BODY = 1,
        FACES_HEADER = 2,
        FACES_BODY = 3,
        IMPORT_DONE = 4,
        IMPORT_ERROR = 5
    };

    ImportState state = VERTICES_HEADER;
    int numVertices;
    int numFaces;

    BasicModel* newModel = new BasicModel();

    ifstream input;
    input.open(filePath.c_str());
    if (input.good() == false) {
        return NULL;
    }

    std::string curLine;

    while (state < IMPORT_DONE && getline(input, curLine)) {
        istringstream lineStream(curLine);
        switch (state) {
        case VERTICES_HEADER:
            if (lineStream >> numVertices) {
                state = VERTICES_BODY;
            }
            break;
        case VERTICES_BODY: {
            Vector3 vertex;
            if (lineStream >> vertex[0] >> vertex[1] >> vertex[2]) {
                newModel->m_vertexArray.push(vertex);
            }
            if (newModel->m_vertexArray.size() == numVertices) {
                state = FACES_HEADER;
            }
            break;
        }
        case FACES_HEADER:
            if (lineStream >> numFaces) {
                state = FACES_BODY;
            }
            break;
        case FACES_BODY: {
            BasicFace face;
            if (lineStream >> face.vertexIndices[0] >>
                    face.vertexIndices[1] >> face.vertexIndices[2]) {
                newModel->m_faceArray.push(face);
            }
            if (newModel->m_faceArray.size() == numFaces) {
                state = IMPORT_DONE;
            }
            break;
        }
        default:
            debugAssert(0);
            break;
        }
    }

    input.close();

    if (state == IMPORT_DONE) {
        newModel->m_isBackfaceArray.resize(newModel->m_faceArray.size());
        newModel->m_extrudedVertexArray.resize(newModel->m_vertexArray.size());
        newModel->computeStaticFaceNormals(newModel->m_vertexArray,
                newModel->m_faceNormalArray);
        newModel->computeBoundingBox();
        newModel->computeBoundingSphere();
        newModel->computeEdges();
        newModel->m_modelColor = Color3::GRAY;
        return newModel;
    } else {
        delete newModel;
        return NULL;
    }
}


bool exportSMFile(
        const BasicModel&       model,
        const std::string&      filePath)
{
    ofstream output;
    output.open(filePath.c_str());
    if (output.good() == false) {
        return false;
    }

    int i;

    output << model.m_vertexArray.size() << endl << endl;
    for (i = 0; i < model.m_vertexArray.size(); i++) {
        const Vector3& curVertex = model.m_vertexArray[i];
        output << curVertex[0] << " " <<
            curVertex[1] << " " <<
            curVertex[2] << " " << endl;
    }

    output << endl << model.m_faceArray.size() << endl << endl;
    for (i = 0; i < model.m_faceArray.size(); i++) {
        const BasicFace& curFace = model.m_faceArray[i];
        output << curFace.vertexIndices[0] << " " << 
            curFace.vertexIndices[1] << " " <<
            curFace.vertexIndices[2] << " " << endl;
    }

    output.close();

    return true;
}

