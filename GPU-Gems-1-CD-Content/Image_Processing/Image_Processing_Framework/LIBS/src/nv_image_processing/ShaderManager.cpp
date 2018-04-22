// ------------------------------------------------------------------
//
// Contents:
//      ShaderManager class
//
// Description:
//      ShaderManager is a simple singleton class that creates 
//      and stores handles to globally used shader programs.
//
// Author:
//      Frank Jargstorff (8/2003)
//
// ------------------------------------------------------------------


//
// Includes
//

#include "ShaderManager.h"
#include "AssertCG.h"


// -----------------------------------------------------------------------------
// ShaderManager class
//

    //
    // Private static data
    //

ShaderManager * ShaderManager::_gpShaderManager = 0;

    
    //
    // Public static data
    //

CGcontext    ShaderManager::gCgContext;

CGprofile    ShaderManager::gVertexIdentityProfile;
CGprogram    ShaderManager::gVertexIdentityShader;
CGparameter  ShaderManager::gVertexIdentityModelView;


    //
    // Construction and destruction
    //

        // Destructor
        //
ShaderManager::~ShaderManager()
{
}

        // initialize
        //
        // Description:
        //      Initializes the shader manager.
        //          This basically creates all the handles stored in
        //      the static variables. Repeated calls to this method
        //      have no effect.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      None
        //
        // Note:
        //      This method will fail if call previous to the OpenGL
        //      initialization.
        //
        void
ShaderManager::initialize()
{
    if (_gpShaderManager == 0)
        _gpShaderManager = new ShaderManager;
}


    //
    // Private methods
    //

        // Default constructor
ShaderManager::ShaderManager()
{
    gCgContext = cgCreateContext();
    CG_ASSERT_NO_ERROR;

    gVertexIdentityProfile = CG_PROFILE_VP20;
                            // Create the vertex identity program that most
                            // image filters will use as vertex shader.
    gVertexIdentityShader = cgCreateProgramFromFile(gCgContext, CG_SOURCE, CG_SOURCE_PATH "VertexIdentity.cg",
                                                 gVertexIdentityProfile, 0, 0);
    CG_ASSERT_NO_ERROR;
    cgGLLoadProgram(gVertexIdentityShader);
    CG_ASSERT_NO_ERROR;

                            // Set the global handle to the vertex identity 
                            // program's model view matrix.
    gVertexIdentityModelView = cgGetNamedParameter(gVertexIdentityShader, "mModelView");
    CG_ASSERT_NO_ERROR;
}

    
