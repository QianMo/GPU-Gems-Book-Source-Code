#ifndef NV_FILE_H
#define NV_FILE_H

/**
 * \file nv_file.h
 *
 * Copyright (C) 2003 NVIDIA Corporation
 * 
 * This file is provided without support, instruction, or implied 
 * warranty of any kind.  NVIDIA makes no guarantee of its fitness
 * for a particular purpose and is not liable under any circumstances
 * for any damages or loss whatsoever arising from the use or 
 * inability to use this file or items derived from it.
 *
 */


//
// Includes
//

#include <stdio.h>
#include <string>

#include <iostream>



/**
 * A thin wrapper class for c-style file operations.
 *
 */
class DECLSPEC_NV_NVB nv_file: public nv_input_output_stream
{
public:
    //
    // Public types
    //

    enum teMode
    {
        READ_MODE,
        WRITE_MODE,
        CLOSED_MODE
    };

    enum teError
    {
        NO_FILE_ERROR,
        READ_ONLY_FILE_ERROR,
        UNKNOWN_FILE_ERROR
    };


    //
    // Construction and destruction
    //

            /// Default constructor.
    nv_file();

            /// Construct from filename.
    nv_file(const char *sFileName, teMode eMode = READ_MODE);

            /// Destructor.
            virtual
   ~nv_file();


   //
   // Public methods
   //

            /// Get the current file's filename.
            std::string
    filename()
            const;

            /// Get the current file's mode.
            teMode
    mode()
            const;

            /// Open a file by name.
            teError
    open(const char * sFilename, teMode eMode = READ_MODE);

            /// Close this file.
            teError
    close();

            /// Reopen the current file.
            teError
    reopen(teMode eMode = READ_MODE);

            /// Read data.
            int
    read(void * pData, int nBytes);

            /// Write data.
            int
    write(const void * pData, int nBytes);


private:
    //
    // Private methods
    //

            /// Get a modestring from the enumerated mode
            static
            const 
            char *
    ModeString(teMode eMode);


    //
    // Private data
    //

    std::string _sFilename;

    FILE *      _hFile;
    teMode      _eMode;
};

#endif // NV_FILE_H
