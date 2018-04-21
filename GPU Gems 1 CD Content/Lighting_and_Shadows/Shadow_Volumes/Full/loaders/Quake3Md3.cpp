/**
  @file Quake3Md3.cpp

  @maintainer Kevin Egan (ktegan@cs.brown.edu)
  @cite Ben Humphrey (digiben@gametutorials.com)

*/

#include <G3DAll.h>
#include "../BasicModel.h"
#include "../Renderer.h"
#include "Quake3Md3.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


enum Models
{
    Lower = 0,
    Upper = 1,
    Head = 2,
    Weapon = 3,
    NumModels
};


enum Animations
{
    // These are the exact names used in the animation.cfg file
    // BOTH - animations involving legs and torso
    // TORSO - animations involving only legs
    // LEGS - animations involving only torso (upper body)

    BOTH_DEATH1 = 0,
    BOTH_DEAD1,
    BOTH_DEATH2,
    BOTH_DEAD2,
    BOTH_DEATH3,
    BOTH_DEAD3,

    TORSO_GESTURE,
    TORSO_ATTACK,
    TORSO_ATTACK2,
    TORSO_DROP,
    TORSO_RAISE,
    TORSO_STAND,
    TORSO_STAND2,

    LEGS_WALKCR,
    LEGS_WALK,
    LEGS_RUN,
    LEGS_BACK,
    LEGS_SWIM,
    LEGS_JUMP,
    LEGS_LAND,
    LEGS_JUMPB,
    LEGS_LANDB,
    LEGS_IDLE,
    LEGS_IDLECR,
    LEGS_TURN,

    NumAnimations
};


struct Q3Md3Vec2
{
    float                       x;
    float                       y;
};


struct Q3Md3Vec3
{
    float                       x;
    float                       y;
    float                       z;
};


struct Q3Md3Header
{
    char                        headerID[4];
    int                         version;
    char                        fileName[68];
    int                         numFrames;
    int                         numTags;
    int                         numMeshes;
    int                         numMaxSkins;
    int                         headerSize;
    int                         tagStart;
    int                         tagEnd;
    int                         fileSize;
};


struct Q3Md3Mesh
{
    char                        meshID[4];
    char                        meshName[68];
    int                         numFrames;
    int                         numSkins;
    int                         numVertices;
    int                         numFaces;
    int                         faceStart;
    int                         headerSize;
    int                         texCoordStart;
    int                         vertexStart;
    int                         meshSize;
};


struct Q3Md3Tag
{
    char                        tagName[64];
    Q3Md3Vec3                   position;
    float                       rotationMatrix[3][3];
};


struct Q3Md3Bone
{
    Q3Md3Vec3                   minBound;
    Q3Md3Vec3                   maxBound;
    Q3Md3Vec3                   position;
    float                       scale;
    char                        creator[16];
};


struct Q3Md3Vertex
{
    short                       position[3];
    unsigned char               normal[2];
};


struct Q3Md3Face
{
    int                         vertexIndices[3];
};


struct Q3Md3TexCoord
{
    float                       textureCoord[2];
};


struct Q3Md3Skin
{
    char                        skinName[68];
};




struct DemMd3Material
{
    std::string                 m_name;
    std::string                 m_fileName;
    Color3                      m_color;
    int                         m_textureIndex;
};


struct DemMd3Tag
{
    std::string                 m_tagName;
    Vector3                     m_position;
    Quat                        m_rotation;
};


struct DemMd3Animation
{
    std::string                 m_animationName;
    int                         m_fps;
    int                         m_startFrame;
    int                         m_endFrame;
};


struct DemMd3Part
{
    std::string                 m_partName;
    bool                        m_useTexture;
    Array<Vector3>              m_vertexArray;
    Array<Vector3>              m_normalArray;
    Array<Vector2>              m_texCoordArray;
    Array<BasicFace>            m_faceArray;
    int                         m_materialIndex;
};


struct DemMd3Model
{
    Array<DemMd3Part>           m_partArray;
    Array<DemMd3Material>       m_materialArray;
    Array<DemMd3Animation>      m_animationArray;

    Array<DemMd3Tag>            m_tagArray;
    Array<DemMd3Model*>         m_linkArray;

    int m_curAnimation;
    int m_curAnimationFrame;
    int m_lastMilliTime;
};


/**
 * Some of the vocabulary is a little confusing:
 * each animating quake character (aka DemMd3Wrapper)
 *   can have several sub-models (ie legs, torso, head)
 * each one of these models can have different parts
 * each part has it's own vertex and texture information
 */
class DemMd3Wrapper : public BasicModel
{
public:
    DemMd3Wrapper();
    virtual ~DemMd3Wrapper(); 

    virtual void updateModel(int milliTime);
    virtual void drawFaces(int& polyCount);
    virtual void drawModel(
        DemMd3Wrapper&              wrapper,
        DemMd3Model&                model,
        int&                        polyCount);

    virtual void useTextures(bool texturesOn);

    Array<DemMd3Model>          m_modelArray;
    Array<TextureRef>           m_textureArray;
    bool                        m_useTextures;

};


/**
 * This calculates face, texture coordinate and vertex index
 * information.  This information is computed once when the model
 * is loaded.
 */
void calculateQuake3StaticData(
        DemMd3Wrapper&          wrapper,
        DemMd3Model&            model)
{
    int i;
    int j;
    int startFace = wrapper.m_faceArray.size();

    for (i = 0; i < model.m_partArray.size(); i++) {
        DemMd3Part& curPart = model.m_partArray[i];
        int newVertices = curPart.m_texCoordArray.size();
        int newFaces = curPart.m_faceArray.size();

        int oldVertices = wrapper.m_texCoordArray.size();
        int oldFaces = wrapper.m_faceArray.size();

        wrapper.m_vertexArray.resize(newVertices + oldVertices);
        wrapper.m_extrudedVertexArray.resize(newVertices + oldVertices);
        wrapper.m_texCoordArray.resize(newVertices + oldVertices);
        wrapper.m_faceArray.resize(newFaces + oldFaces);
        wrapper.m_isBackfaceArray.resize(newFaces + oldFaces);

        for (j = 0; j < newVertices; j++) {
            wrapper.m_texCoordArray[j + oldVertices] =
                curPart.m_texCoordArray[j];
        }

        for (j = 0; j < newFaces; j++) {
            wrapper.m_faceArray[j + oldFaces] = curPart.m_faceArray[j];
            wrapper.m_faceArray[j + oldFaces].vertexIndices[0] += oldVertices;
            wrapper.m_faceArray[j + oldFaces].vertexIndices[1] += oldVertices;
            wrapper.m_faceArray[j + oldFaces].vertexIndices[2] += oldVertices;
        }
    }

    for (i = 0; i < model.m_linkArray.size(); i++) {
        if (model.m_linkArray[i] != NULL) {
            calculateQuake3StaticData(wrapper, *(model.m_linkArray[i]));
        }
    }
}


/**
 * This updates vertex data depending on the current animation
 * frame and the coordinate system of the model we are in
 * (ie the torso may have a slightly different coordinate system
 * then the legs).  This must be recomputed every time we want
 * to move vertices for animation.  Note that this does
 * not interpolate between frames.
 */
void calculateQuake3DynamicData(
        DemMd3Wrapper&          wrapper,
        DemMd3Model&            model,
        int                     curVertex,
        CoordinateFrame&        curCoordFrame)
{
    int i;
    int j;

    for (i = 0; i < model.m_partArray.size(); i++) {
        DemMd3Part& curPart = model.m_partArray[i];

        int newVertices = curPart.m_texCoordArray.size();

        for (j = 0; j < newVertices; j++) {
            int animationIndex = j + curPart.m_texCoordArray.size() *
                model.m_curAnimationFrame;
            wrapper.m_vertexArray[j + curVertex] =
                curCoordFrame.pointToWorldSpace(
                        curPart.m_vertexArray[animationIndex]);
        }
        curVertex += newVertices;
    }

    for (i = 0; i < model.m_linkArray.size(); i++) {
        if (model.m_linkArray[i] != NULL) {
            int curTagIndex = i + model.m_curAnimationFrame *
                model.m_linkArray.size();
            
            CoordinateFrame childTransform(
                    model.m_tagArray[curTagIndex].m_rotation.toRotationMatrix(),
                    model.m_tagArray[curTagIndex].m_position);

            calculateQuake3DynamicData(wrapper,
                    *(model.m_linkArray[i]), curVertex,
                    curCoordFrame * childTransform);
        }
    }
}


/**
 * Because quake 3 md3 files are made in separate models the best we
 * can hope for is that each model is a closed surface.  For this
 * reason we calculate edges on each of the models separately.  This
 * is necessary because some collapsing happens during the edge
 * calculation, and if we do all of the models at once vertices from
 * two different models may be collapsed together and then finding
 * edge information becomes impossible.
 */
void calculateQuake3Edges(
        DemMd3Wrapper&          wrapper,
        DemMd3Model&            model,
        int                     startFace)
{
    int endFace = startFace;
    int i;

    for (i = 0; i < model.m_partArray.size(); i++) {
        DemMd3Part& curPart = model.m_partArray[i];
        endFace += curPart.m_faceArray.size();
    }

    wrapper.computeEdges(startFace, endFace);

    for (i = 0; i < model.m_linkArray.size(); i++) {
        if (model.m_linkArray[i] != NULL) {
            calculateQuake3Edges(wrapper,
                *(model.m_linkArray[i]), endFace);
        }
    }
}


DemMd3Wrapper::DemMd3Wrapper()
{
    m_boundingBox = NULL;
    m_modelColor = Color3(1.0, 1.0, 1.0);
}


DemMd3Wrapper::~DemMd3Wrapper()
{
    // do nothing
}


void DemMd3Wrapper::useTextures(bool texturesOn)
{
    m_useTextures = texturesOn;
}


/**
 * This advances one frame in an animation.  Note that this does
 * not interpolate between frames.
 */
void DemMd3Wrapper::updateModel(int milliTime)
{
    int i;

    for (i = 0; i < m_modelArray.size(); i++) {
        DemMd3Model& model = m_modelArray[i];
        if ((model.m_animationArray.size() > 0) &&
                (milliTime - model.m_lastMilliTime > 50)) {
            model.m_curAnimationFrame++;
            if (model.m_curAnimationFrame ==
                model.m_animationArray[model.m_curAnimation].m_endFrame) {

                model.m_curAnimation++;
                model.m_curAnimation %= model.m_animationArray.size();

                model.m_curAnimationFrame =
                    model.m_animationArray[model.m_curAnimation].m_startFrame;
            }
            debugAssert(model.m_curAnimationFrame <
                    model.m_partArray[0].m_vertexArray.size() /
                    model.m_partArray[0].m_texCoordArray.size());
            model.m_lastMilliTime = milliTime;
        }
    }

    // convert from id's coordinate system to GL's coordinate system
    Matrix3 quakeRotation(Matrix3::ZERO);
    quakeRotation.fromAxisAngle(Vector3(1, 0, 0), -G3D_PI / 2.0);
    CoordinateFrame toQuakeAxes(quakeRotation, Vector3(0, 0, 0));

    calculateQuake3DynamicData(*this, m_modelArray[Lower], 0, toQuakeAxes);
    computeStaticFaceNormals(m_vertexArray, m_faceNormalArray);

    m_extrusionDirty = true;
    computeBoundingBox();
    computeBoundingSphere();
}


/**
 * This is a recursive procedure that starts at the legs and
 * draws all of the different "sub-models" for the DemMd3Wrapper.
 */
void DemMd3Wrapper::drawModel(
        DemMd3Wrapper&              wrapper,
        DemMd3Model&                model,
        int&                        polyCount)
{
    int i;
    int j;

    for (i = 0; i < model.m_partArray.size(); i++) {
        DemMd3Part& curPart = model.m_partArray[i];
        if (m_useTextures && curPart.m_useTexture) {
            glEnable(GL_TEXTURE_2D);
            DemMd3Material& material =
                model.m_materialArray[curPart.m_materialIndex];
            if (material.m_textureIndex >= 0) {
                TextureRef& texture = m_textureArray[material.m_textureIndex];
                glBindTexture(GL_TEXTURE_2D, texture->getOpenGLID());
                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
            } else {
                glDisable(GL_TEXTURE_2D);
            }
        } else {
            glDisable(GL_TEXTURE_2D);
        }

        int endFace = polyCount + curPart.m_faceArray.size();
        glBegin(GL_TRIANGLES);
            for (j = polyCount; j < endFace; j++) {
                glNormal(wrapper.m_faceNormalArray[j]);

                int vertexIndex = wrapper.m_faceArray[j].vertexIndices[0];
                glTexCoord(wrapper.m_texCoordArray[vertexIndex]);
                glVertex(wrapper.m_vertexArray[vertexIndex]);

                vertexIndex = wrapper.m_faceArray[j].vertexIndices[1];
                glTexCoord(wrapper.m_texCoordArray[vertexIndex]);
                glVertex(wrapper.m_vertexArray[vertexIndex]);

                vertexIndex = wrapper.m_faceArray[j].vertexIndices[2];
                glTexCoord(wrapper.m_texCoordArray[vertexIndex]);
                glVertex(wrapper.m_vertexArray[vertexIndex]);
            }
        glEnd();
        polyCount += curPart.m_faceArray.size();
    }

    for (i = 0; i < model.m_linkArray.size(); i++) {
        DemMd3Model* child = model.m_linkArray[i];
        if (child != NULL) {
            int curTagIndex = i + model.m_curAnimationFrame *
                model.m_linkArray.size();
            drawModel(wrapper, *(child), polyCount);
        }
    }
}


void DemMd3Wrapper::drawFaces(
        int&                    polyCount)
{
    polyCount = 0;

    // we may turn on texturing so call glPushAttrib()
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    drawModel(*this, m_modelArray[Lower], polyCount);
    glPopAttrib();
}


/**
 * This loads one model.  However in this context a model is not
 * the whole animating character, rather it is the character's legs,
 * torso or head most likely.
 */
void loadQuake3Model(
        DemMd3Model &           model,
        const std::string &     fileName)
{
    FILE* fp;
    int i;
    int j;

    static const float scaleFactor = 0.1;

    fp = fopen(fileName.c_str(), "rb");
    debugAssertM(fp != NULL,
            std::string("could not open md3 file ") + fileName);

    Q3Md3Header header;
    fread(& header, sizeof(header), 1, fp);

    debugAssertM(strncmp("IDP3", header.headerID, 4) == 0,
            std::string("Q3Md3 file has invalid ID ") + fileName);
    debugAssertM(header.version == 15,
            std::string("Q3Md3 file has invalid version ") + fileName);

    Q3Md3Bone* boneData = new Q3Md3Bone[header.numFrames];
    fread(boneData, sizeof(boneData[0]), header.numFrames, fp);

    int totalTags = header.numTags * header.numFrames;
    Q3Md3Tag* tagData = new Q3Md3Tag[totalTags];
    fread(tagData, sizeof(tagData[0]), totalTags, fp);

    model.m_tagArray.resize(totalTags);
    for (i = 0; i < totalTags; i++) {
        model.m_tagArray[i].m_tagName = tagData[i].tagName;

        const Q3Md3Vec3 & pos = tagData[i].position;
        model.m_tagArray[i].m_position =
            Vector3(pos.x, pos.y, pos.z) * scaleFactor;

        Matrix3 mat(tagData[i].rotationMatrix);
        model.m_tagArray[i].m_rotation = Quat(mat.transpose());
    }

    model.m_linkArray.resize(header.numTags);
    for (i = 0; i < header.numTags; i++) {
        model.m_linkArray[i] = NULL;
    }

    delete[] boneData;
    delete[] tagData;


    int fileOffset = ftell(fp);

    model.m_partArray.resize(header.numMeshes);

    for (i = 0; i < header.numMeshes; i++) {
        Q3Md3Mesh mesh;
        fseek(fp, fileOffset, SEEK_SET);
        fread(& mesh, sizeof(mesh), 1, fp);

        // allocate memory
        Q3Md3Vertex* vertexData = new Q3Md3Vertex[mesh.numVertices *
            header.numFrames];
        Q3Md3TexCoord* texCoordData = new Q3Md3TexCoord[mesh.numVertices];
        Q3Md3Face* faceData = new Q3Md3Face[mesh.numFaces];
        Q3Md3Skin* skinData = new Q3Md3Skin[mesh.numSkins];

        // read in data
        fread(skinData, sizeof(skinData[0]), mesh.numSkins, fp);

        fseek(fp, fileOffset + mesh.vertexStart, SEEK_SET);
        fread(vertexData, sizeof(vertexData[0]), mesh.numVertices *
                header.numFrames, fp);

        fseek(fp, fileOffset + mesh.faceStart, SEEK_SET);
        fread(faceData, sizeof(faceData[0]), mesh.numFaces, fp);

        fseek(fp, fileOffset + mesh.texCoordStart, SEEK_SET);
        fread(texCoordData, sizeof(texCoordData[0]), mesh.numVertices, fp);


        // store data in final model
        DemMd3Part& part = model.m_partArray[i];
        part.m_partName = mesh.meshName;

        part.m_vertexArray.resize(mesh.numVertices * header.numFrames);
        part.m_normalArray.resize(mesh.numVertices * header.numFrames);
        for (j = 0; j < mesh.numVertices * header.numFrames; j++) {
            const Q3Md3Vertex & vert = vertexData[j];
            part.m_vertexArray[j] = Vector3(vert.position[0] / 64.0,
                    vert.position[1] / 64.0, vert.position[2] / 64.0) *
                    scaleFactor;

            double longitude = vert.normal[0] * (2 * G3D_PI) / 255.0;
            double lattitude = vert.normal[1] * (2 * G3D_PI) / 255.0;
            part.m_normalArray[j] = Vector3(cos(lattitude) * sin(longitude),
                cos(longitude), -sin(lattitude) * sin(longitude));
        }

        part.m_texCoordArray.resize(mesh.numVertices);
        for (j = 0; j < mesh.numVertices; j++) {
            part.m_texCoordArray[j][0] = texCoordData[j].textureCoord[0];
            part.m_texCoordArray[j][1] = texCoordData[j].textureCoord[1];
        }

        part.m_faceArray.resize(mesh.numFaces);
        for (j = 0; j < mesh.numFaces; j++) {
            part.m_faceArray[j].vertexIndices[0] = faceData[j].vertexIndices[0];
            part.m_faceArray[j].vertexIndices[1] = faceData[j].vertexIndices[2];
            part.m_faceArray[j].vertexIndices[2] = faceData[j].vertexIndices[1];
        }

        fileOffset += mesh.meshSize;
    }
}


/**
 * This loads structs that say what textures are needed for what models
 */
void loadQuake3Skins(
        DemMd3Model&            model,
        const std::string &     fileName)
{
    int i;
    std::ifstream input(fileName.c_str());

    debugAssertM(input.good(), std::string("could not open file ") + fileName);

    for (i = 0; i < model.m_partArray.size(); i++) {
        model.m_partArray[i].m_useTexture = false;
        model.m_partArray[i].m_materialIndex = -1;
    }

    std::string curline;
    while (std::getline(input, curline)) {
        for (i = 0; i < model.m_partArray.size(); i++) {
            DemMd3Part & part = model.m_partArray[i];
            int namePos = curline.find(part.m_partName);
            int slashPos = curline.rfind('/');
            if (namePos != std::string::npos) {
                std::string texName;
                if (slashPos != std::string::npos) {
                    texName = curline.substr(slashPos + 1, curline.length());
                }

                model.m_materialArray.resize(model.m_materialArray.size() + 1);
                DemMd3Material& material = model.m_materialArray[
                    model.m_materialArray.size() - 1];
                material.m_name = part.m_partName;
                material.m_fileName = texName;
                material.m_textureIndex = -1;

                part.m_useTexture = true;
                part.m_materialIndex = model.m_materialArray.size() - 1;
            }
        }
    }

    input.close();
    
}


/**
 * This loads animation structs which say which sets of
 * vertices/indices belong to which animations.
 */
void loadQuake3Animations(
        Array<DemMd3Model> &    modelArray,
        const std::string &     fileName)
{
    std::ifstream input(fileName.c_str());
    debugAssertM(input.good(),
            std::string("could not open animation file ") + fileName);

    std::string curline;
    int torsoOffset = 0;
    int legFrameFix = -1;

    while (std::getline(input, curline)) {
        int startFrame;
        int frameCount;
        int loopingFrames;
        int fps;
        std::string separator;
        std::string animationName;

        std::istringstream lineStream(curline);

        if ((lineStream >> startFrame >> frameCount >> loopingFrames >> fps >>
                separator >> animationName) &&
                (separator == std::string("//"))) {

            int splitPos = animationName.find('_');
            debugAssert(splitPos != std::string::npos);
            std::string model = animationName.substr(0, splitPos);

            DemMd3Animation animation;
            animation.m_animationName = animationName;
            animation.m_fps = fps;
            animation.m_startFrame = startFrame;
            animation.m_endFrame = startFrame + frameCount;

            if (model == std::string("BOTH")) {
                modelArray[Lower].m_animationArray.push(animation);
                modelArray[Upper].m_animationArray.push(animation);
            } else if (model == std::string("TORSO")) {
                modelArray[Upper].m_animationArray.push(animation);
            } else if (model == std::string("LEGS")) {
                if (legFrameFix == -1) {
                    const DemMd3Animation& firstTorsoAnimation =
                        modelArray[Upper].m_animationArray[TORSO_GESTURE];
                    legFrameFix = animation.m_startFrame -
                        firstTorsoAnimation.m_startFrame;
                    debugAssert(legFrameFix >= 0);
                }
                animation.m_startFrame -= legFrameFix;
                animation.m_endFrame -= legFrameFix;
                modelArray[Lower].m_animationArray.push(animation);
            } else {
                debugAssertM(false, std::string("unkown model ") +
                    model + std::string(" in ") + fileName);
            }
        }
    }

    input.close();
}


void loadQuake3Textures(
        DemMd3Model &           model,
        Array<TextureRef> &     textureArray,
        const std::string &     path)
{
    int i;
    int j;

    for (i = 0; i < model.m_materialArray.size(); i++) {
        if (model.m_materialArray[i].m_fileName.empty() == false &&
                model.m_materialArray[i].m_fileName != "nodraw") {
            bool alreadyLoaded = false;
            
            for (j = 0; j < textureArray.size(); j++) {
                if (model.m_materialArray[i].m_fileName ==
                        textureArray[j]->getName()) {
                    alreadyLoaded = true;
                    model.m_materialArray[i].m_textureIndex = j;
                    break;
                }
            }

            if (!alreadyLoaded) {
                //textureArray.push(new Texture(
                //        model.m_materialArray[i].m_name,
                //        path + model.m_materialArray[i].m_fileName, ""));
				textureArray.push(Texture::fromFile(path + model.m_materialArray[i].m_fileName));
                model.m_materialArray[i].m_textureIndex =
                        textureArray.size() - 1;
            }
        }
    }
}


/**
 * fill out m_linkArray pointers now that we've read in all of the
 * models for one wrapper
 */
void linkQuake3Models(
        DemMd3Model &           parent,
        DemMd3Model *           child,
        const std::string &     tagName)
{
    int i;

    for (i = 0; i < parent.m_tagArray.size(); i++) {
        if (parent.m_tagArray[i].m_tagName == tagName) {
            parent.m_linkArray[i] = child;
            break;
        }
    }
}


/**
 * This creates a sub-class of the BasicModel object that overrides
 * update() to do animation.
 */
BasicModel* loadQuake3Md3(
        const std::string &     modelName,
        const std::string &     path,
        int                     startAnimation)
{
    int i;
    std::string modelExtension[3];

    modelExtension[Lower] = "lower";
    modelExtension[Upper] = "upper";
    modelExtension[Head] = "head";

    DemMd3Wrapper* wrapper = new DemMd3Wrapper();
    wrapper->m_modelArray.resize(NumModels);

    for (i = 0; i < 3; i++) {
        std::string finalPath = path + modelExtension[i] + ".md3";
        
        loadQuake3Model(wrapper->m_modelArray[i], finalPath);
    }

    // load skins
    for (i = 0; i < 3; i++) {
        std::string finalPath = path + modelExtension[i] + "_" +
            modelName + ".skin";
        loadQuake3Skins(wrapper->m_modelArray[i], finalPath);
    }
    
    // load animations
    loadQuake3Animations(wrapper->m_modelArray, path + "animation.cfg");
    
    // initialize model animation info
    for (i = 0; i < 3; i++) {
        DemMd3Model& model = wrapper->m_modelArray[i];
        model.m_lastMilliTime = -1;
        if (model.m_animationArray.size() > 0) {
            model.m_curAnimation = startAnimation % model.m_animationArray.size();
            model.m_curAnimationFrame =
                model.m_animationArray[model.m_curAnimation].m_startFrame;
        } else {
            model.m_curAnimation = 0;
            model.m_curAnimationFrame = 0;
        }
    }

    // load textures
    for (i = 0; i < 3; i++) {
        loadQuake3Textures(wrapper->m_modelArray[i],
                wrapper->m_textureArray, path);
    }

    // link models
    linkQuake3Models(wrapper->m_modelArray[Lower],
        & wrapper->m_modelArray[Upper], "tag_torso");
    linkQuake3Models(wrapper->m_modelArray[Upper],
        & wrapper->m_modelArray[Head], "tag_head");

    CoordinateFrame identity;
    calculateQuake3StaticData(*wrapper, wrapper->m_modelArray[Lower]);
    calculateQuake3DynamicData(*wrapper, wrapper->m_modelArray[Lower],
            0, identity);
    calculateQuake3Edges(*wrapper, wrapper->m_modelArray[Lower], 0);

    return wrapper;
}


