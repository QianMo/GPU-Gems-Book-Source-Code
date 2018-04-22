#ifndef SHADER_MANAGER_H
#define SHADER_MANAGER_H
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

#include <Cg/cg.h>
#include <Cg/cgGL.h>


//
// Constant defines
//

#define CG_SOURCE_PATH "../../../../MEDIA/programs/nv_image_processing/"


// -----------------------------------------------------------------------------
// ShaderManager class
//
class ShaderManager
{
public:
    //
    // Construction and destruction
    //

            // Destructor
            //
   ~ShaderManager();

    //
    // Public static methods
    //

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
            static
            void
    initialize();


    // 
    // Public static data
    //

    static CGcontext    gCgContext;

    static CGprofile    gVertexIdentityProfile;
    static CGprogram    gVertexIdentityShader;
    static CGparameter  gVertexIdentityModelView;


private:
    //
    // Private methods
    //

            // Default constructor
    ShaderManager();

            // Copy constructor (not implemented)
    ShaderManager(const ShaderManager &);

            // Assignment operator (not implemented)
    operator=(const ShaderManager &);
    
    //
    // Private static data
    //

    static ShaderManager * _gpShaderManager;
};

#endif // SHADER_MANAGER_H
