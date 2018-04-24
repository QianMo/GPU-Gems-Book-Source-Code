/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:   
*
* This source code is subject to NVIDIA ownership rights under U.S. and 
* international Copyright laws.  
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
* OR PERFORMANCE OF THIS SOURCE CODE.  
*
* U.S. Government End Users.  This source code is a "commercial item" as 
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
* "commercial computer software" and "commercial computer software 
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
* and is provided to the U.S. Government only as a commercial end item.  
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
* source code with only those rights set forth herein.
*/


/* CUda UTility Library */

/* Credit: Cuda team for the PGM file reader / writer code. */

// includes, file
#include <cutil.h>

// includes, system
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>

// includes, cuda
#include <builtin_types.h>

// includes, common
#include <cmd_arg_reader.h>
#include <error_checker.h>
#include <stopwatch.h>
#include <bank_checker.h>

// includes, system

// namespace unnamed (internal)
namespace 
{  
    // variables

    //! size of PGM file header 
    const unsigned int PGMHeaderSize = 0x40;


    // types

    //! Data converter from unsigned char / unsigned byte to type T
    template<class T>
    struct ConverterFromUByte;

    //! Data converter from unsigned char / unsigned byte to unsigned int
    template<>
    struct ConverterFromUByte<unsigned int> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned int operator()( const unsigned char& val) {
            return static_cast<unsigned int>( val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned short
    template<>
    struct ConverterFromUByte<unsigned short> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned short operator()( const unsigned char& val) 
        {
            return static_cast<unsigned short>( val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned float
    template<>
    struct ConverterFromUByte<float> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        float operator()( const unsigned char& val) 
        {
            return static_cast<float>( val) / 255.0f;
        }
    };

    //! Data converter from unsigned char / unsigned byte to type T
    template<class T>
    struct ConverterToUByte;

    //! Data converter from unsigned char / unsigned byte to unsigned int
    template<>
    struct ConverterToUByte<unsigned int> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()( const unsigned int& val) 
        {
            return static_cast<unsigned char>( val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned short
    template<>
    struct ConverterToUByte<unsigned short> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()( const unsigned short& val) 
        {
            return static_cast<unsigned char>( val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned float
    template<>
    struct ConverterToUByte<float> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()( const float& val) 
        {
            return static_cast<unsigned char>( val * 255.0f);
        }
    };


    // functions

    //////////////////////////////////////////////////////////////////////////////
    //! Load PGM or PPM file
    //! @return CUTTrue if the file loading succeeded, otherwise false
    //! @param file        name of the file to load
    //! @param data        handle to the memory for the image file data
    //! @param w        width of the image
    //! @param h        height of the image
    //! @param channels number of channels in image
    //////////////////////////////////////////////////////////////////////////////
    CUTBoolean
    loadPPM( const char* file, unsigned char** data, 
             unsigned int *w, unsigned int *h, unsigned int *channels ) 
    {
        FILE *fp = NULL;
        if(NULL == (fp = fopen(file, "rb"))) 
        {
            std::cerr << "cutLoadPPM() : Failed to open file: " << file << std::endl;
            return CUTFalse;
        }

        // check header
        char header[PGMHeaderSize];
        fgets( header, PGMHeaderSize, fp);
        if (strncmp(header, "P5", 2) == 0)
        {
            *channels = 1;
        }
        else if (strncmp(header, "P6", 2) == 0)
        {
            *channels = 3;
        }
        else {
            std::cerr << "cutLoadPPM() : File is not a PPM or PGM image" << std::endl;
            *channels = 0;
            return CUTFalse;
        }

        // parse header, read maxval, width and height
        unsigned int width = 0;
        unsigned int height = 0;
        unsigned int maxval = 0;
        unsigned int i = 0;
        while(i < 3) 
        {
            fgets(header, PGMHeaderSize, fp);
            if(header[0] == '#') 
                continue;

            if(i == 0) 
            {
                i += sscanf( header, "%u %u %u", &width, &height, &maxval);
            }
            else if (i == 1) 
            {
                i += sscanf( header, "%u %u", &height, &maxval);
            }
            else if (i == 2) 
            {
                i += sscanf(header, "%u", &maxval);
            }
        }

        // check if given handle for the data is initialized
        if( NULL != *data) 
        {
            if (*w != width || *h != height) 
            {
                std::cerr << "cutLoadPPM() : Invalid image dimensions." << std::endl;
                return CUTFalse;
            }
        } 
        else 
        {
            *data = new unsigned char[width * height * *channels];
            *w = width;
            *h = height;
        }

        // read adn close file
        fread( *data, sizeof(unsigned char), width * height * *channels, fp);
        fclose(fp);

        return CUTTrue;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Generic PGM image file loader adapter for type T
    //! @return CUTTrue if the file loading succeeded, otherwise false
    //! @param file  name of the file to load
    //! @param data  handle to the memory for the image file data
    //! @param w     width of the image
    //! @param h     height of the image
    //////////////////////////////////////////////////////////////////////////////
    template<class T>
    CUTBoolean
    loadPGMt( const char* file, T** data, unsigned int *w, unsigned int *h) 
    {
        unsigned char* idata = NULL;
        unsigned int channels;
        if( CUTTrue != loadPPM(file, &idata, w, h, &channels)) 
        {
            return CUTFalse;
        }

        unsigned int size = *w * *h * channels;

        // initialize mem if necessary
        // the correct size is checked / set in loadPGMc()
        if( NULL == *data) 
        {
            *data = new T[size];
        }

        // copy and cast data
        std::transform( idata, idata + size, *data, ConverterFromUByte<T>());

        delete[] idata;

        return CUTTrue;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Write / Save PPM or PGM file
    //! @note Internal usage only
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    //////////////////////////////////////////////////////////////////////////////  
    CUTBoolean
    savePPM( const char* file, unsigned char *data, 
             unsigned int w, unsigned int h, unsigned int channels) 
    {
        CUT_CONDITION( NULL != data);
        CUT_CONDITION( w > 0);
        CUT_CONDITION( h > 0);

        std::fstream fh( file, std::fstream::out | std::fstream::binary );
        if( fh.bad()) 
        {
            std::cerr << "savePPM() : Opening file failed." << std::endl;
            return CUTFalse;
        }

        if (channels == 1)
        {
            fh << "P5\n";
        }
        else if (channels == 3) {
            fh << "P6\n";
        }
        else {
            std::cerr << "savePPM() : Invalid number of channels." << std::endl;
            return CUTFalse;
        }

        fh << w << "\n" << h << "\n" << 0xff << std::endl;

        for( unsigned int i = 0; (i < (w*h*channels)) && fh.good(); ++i) 
        {
            fh << data[i];
        }
        fh.flush();

        if( fh.bad()) 
        {
            std::cerr << "savePPM() : Writing data failed." << std::endl;
            return CUTFalse;
        } 
        fh.close();

        return CUTTrue;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Generic PGM file writer for input data of type T
    //! @note Internal usage only
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    //////////////////////////////////////////////////////////////////////////////
    template<class T>
    CUTBoolean
    savePGMt( const char* file, T *data, unsigned int w, unsigned int h) 
    {
        unsigned int size = w * h;
        unsigned char* idata = new unsigned char[size];

        std::transform( data, data + size, idata, ConverterToUByte<T>());

        // write file
        CUTBoolean result = savePPM(file, idata, w, h, 1);

        // cleanup
        delete[] idata;

        return result;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Generic PPM file writer for input data of type T
    //! @note Internal usage only
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    //////////////////////////////////////////////////////////////////////////////
    template<class T>
    CUTBoolean
    savePPMt( const char* file, T *data, unsigned int w, unsigned int h) 
    {
        unsigned int size = w * h * 3;
        unsigned char* idata = new unsigned char[size];

        std::transform( data, data + size, idata, ConverterToUByte<T>());

        // write file
        CUTBoolean result = savePPM(file, idata, w, h, 3);

        // cleanup
        delete[] idata;

        return result;
    }

    ////////////////////////////////////////////////////////////////////////////// 
    //! Compare two arrays of arbitrary type       
    //! @return  true if \a reference and \a data are identical, otherwise false
    //! @param reference  handle to the reference data / gold image
    //! @param data       handle to the computed data
    //! @param len        number of elements in reference and data
    //! @param epsilon    epsilon to use for the comparison
    //////////////////////////////////////////////////////////////////////////////
    template<class T, class S>
    CUTBoolean  
    compareData( const T* reference, const T* data, const unsigned int len, 
                 const S epsilon) 
    {
        CUT_CONDITION( epsilon >= 0);

        bool result = true;

        for( unsigned int i = 0; i < len; ++i) {

            T diff = reference[i] - data[i];
            bool comp = (diff <= epsilon) && (diff >= -epsilon);
            result &= comp;

#ifdef _DEBUG
            if( ! comp) 
            {
                std::cerr << "ERROR, i = " << i << ",\t " 
                    << reference[i] << " / "
                    << data[i] 
                    << " (reference / data)\n";
            }
#endif
        }

        return (result) ? CUTTrue : CUTFalse;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Read file \filename and return the data
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param filename name of the source file
    //! @param data  uninitialized pointer, returned initialized and pointing to
    //!        the data read
    //! @param len  number of data elements in data, -1 on error
    //////////////////////////////////////////////////////////////////////////////
    template<class T>
    CUTBoolean
    cutReadFile( const char* filename, T** data, unsigned int* len, bool verbose) 
    {
        // check input arguments
        CUT_CONDITION( NULL != filename);
        CUT_CONDITION( NULL != len);

        // intermediate storage for the data read
        std::vector<T>  data_read;

        // open file for reading
        std::fstream fh( filename, std::fstream::in);
        // check if filestream is valid
        if( ! fh.good()) 
        {
            if (verbose)
                std::cerr << "cutReadFile() : Opening file failed." << std::endl;
            return CUTFalse;
        }

        // read all data elements 
        T token;
        while( fh.good()) 
        {
            fh >> token;   
            data_read.push_back( token);
        }

        // the last element is read twice
        data_read.pop_back();

        // check if reading result is consistent
        if( ! fh.eof()) 
        {
            if (verbose)
                std::cerr << "WARNING : readData() : reading file might have failed." 
                << std::endl;
        }

        fh.close();

        // check if the given handle is already initialized
        if( NULL != *data) 
        {
            if( *len != data_read.size()) 
            {
                if (verbose)
                    std::cerr << "cutReadFile() : Initialized memory given but size "
                    << " mismatch with signal read." << std::endl;

                return CUTFalse;
            }
        }
        else 
        {
            // allocate storage for the data read
        *data = (T*) malloc(sizeof(T) * data_read.size());  //new T[data_read.size()];
            // store signal size
            *len = static_cast<unsigned int>( data_read.size());
        }

        // copy data
        memcpy( *data, &data_read.front(), sizeof(T) * data_read.size());

        return CUTTrue;
    }

    //////////////////////////////////////////////////////////////////////////////
    //! Write a data file \filename 
    //! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
    //! @param filename name of the source file
    //! @param data  data to write
    //! @param len  number of data elements in data, -1 on error
    //! @param epsilon  epsilon for comparison
    //////////////////////////////////////////////////////////////////////////////
    template<class T>
    CUTBoolean
    cutWriteFile( const char* filename, const T* data, unsigned int len,
                  const T epsilon, bool verbose) 
    {
        CUT_CONDITION( NULL != filename);
        CUT_CONDITION( NULL != data);

        // open file for writing
        std::fstream fh( filename, std::fstream::out);
        // check if filestream is valid
        if( ! fh.good()) 
        {
            if (verbose)
                std::cerr << "cutWriteFile() : Opening file failed." << std::endl;
            return CUTFalse;
        }

        // first write epsilon
        fh << "# " << epsilon << "\n";

        // write data
        for( unsigned int i = 0; (i < len) && (fh.good()); ++i) 
        {
            fh << data[i] << ' ';
        }

        // Check if writing succeeded
        if( ! fh.good()) 
        {
            if (verbose)
                std::cerr << "cutWriteFile() : Writing file failed." << std::endl;
            return CUTFalse;
        }

        // file ends with nl
        fh << std::endl;

        return CUTTrue;
    }

} // end, namespace unnamed
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//! Deallocate memory allocated within Cutil
////////////////////////////////////////////////////////////////////////////////
DLL_MAPPING
void
cutFree( void* ptr) 
{
    if (ptr)
        free( ptr);
}

//////////////////////////////////////////////////////////////////////////////
//! Find the path for a filename
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
char*
cutFindFilePath(const char* filename, const char* executable_path) 
{
    if (filename == 0)
        return 0;
    size_t filename_len = strlen(filename);
    const char data_folder[] = "data/";
    size_t data_folder_len = strlen(data_folder);
    char* file_path = (char*)malloc(data_folder_len + filename_len + 1);
    strcpy(file_path, data_folder);
    strcat(file_path, filename);
    std::fstream fh0(file_path, std::fstream::in);
    if (fh0.good())
        return file_path;
    free(file_path);
    if (executable_path == 0)
        return 0;
    size_t executable_path_len = strlen(executable_path);
    const char* exe;
    for (exe = executable_path + executable_path_len - 1; exe >= executable_path; --exe)
        if (*exe == '/' || *exe == '\\')
            break;
    const char projects_relative_path[] = "/../../../projects";
    size_t projects_relative_path_len = strlen(projects_relative_path);
    file_path = (char*)malloc(executable_path_len + projects_relative_path_len + 1 + data_folder_len + filename_len + 1);
    char* fp = file_path;
    const char* ep = executable_path;
    for (; ep < exe; ++fp, ++ep)
        *fp = *ep;
    *fp = '\0';
    strcat(file_path, projects_relative_path);
    strcat(file_path, exe);
    size_t file_path_len = strlen(file_path);
    if (*(file_path + file_path_len - 1) == 'e' &&
        *(file_path + file_path_len - 2) == 'x' &&
        *(file_path + file_path_len - 3) == 'e' &&
        *(file_path + file_path_len - 4) == '.') {
        *(file_path + file_path_len - 4) = '/';
        *(file_path + file_path_len - 3) = '\0';
    }
    else {
        *(file_path + file_path_len - 0) = '/';
        *(file_path + file_path_len + 1) = '\0';
    }
    strcat(file_path, data_folder);
    strcat(file_path, filename);
    std::fstream fh1(file_path, std::fstream::in);
    if (fh1.good())
        return file_path;
    free(file_path);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg single precision floating point data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutReadFilef( const char* filename, float** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg double precision floating point data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutReadFiled( const char* filename, double** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg integer data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutReadFilei( const char* filename, int** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg unsigned integer data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutReadFileui( const char* filename, unsigned int** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}


////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg char / byte data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutReadFileb( const char* filename, char** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg unsigned char / byte data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutReadFileub( const char* filename, unsigned char** data, unsigned int* len, bool verbose) 
{
    return cutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for single precision floating point data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutWriteFilef( const char* filename, const float* data, unsigned int len,
               const float epsilon, bool verbose) 
{
    return cutWriteFile( filename, data, len, epsilon, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for double precision floating point data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutWriteFiled( const char* filename, const double* data, unsigned int len,
               const double epsilon, bool verbose) 
{
    return cutWriteFile( filename, data, len, epsilon, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for integer data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutWriteFilei( const char* filename, const int* data, unsigned int len, bool verbose) 
{
    return cutWriteFile( filename, data, len, 0, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for unsigned integer data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutWriteFileui( const char* filename,const unsigned int* data,unsigned int len, bool verbose)
{
    return cutWriteFile( filename, data, len, static_cast<unsigned int>(0), verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for byte / char data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutWriteFileb( const char* filename, const char* data, unsigned int len, bool verbose) 
{  
    return cutWriteFile( filename, data, len, static_cast<char>(0), verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for byte / char data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutWriteFileub( const char* filename, const unsigned char* data, 
                unsigned int len, bool verbose) 
{  
    return cutWriteFile( filename, data, len, static_cast<unsigned char>(0), verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for unsigned byte / char data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutWriteFileb( const char* filename,const unsigned char* data,unsigned int len, bool verbose)
{
    return cutWriteFile( filename, data, len, static_cast<unsigned char>(0), verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Load PGM image file (with unsigned char as data element type)
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutLoadPGMub( const char* file, unsigned char** data, 
              unsigned int *w,unsigned int *h)
{
    unsigned int channels;
    return loadPPM( file, data, w, h, &channels);
}

////////////////////////////////////////////////////////////////////////////////
//! Load PPM image file (with unsigned char as data element type)
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutLoadPPMub( const char* file, unsigned char** data, 
              unsigned int *w,unsigned int *h)
{
    unsigned int channels;
    return loadPPM( file, data, w, h, &channels);
}

////////////////////////////////////////////////////////////////////////////////
//! Load PPM image file (with unsigned char as data element type), padding 4th component
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutLoadPPM4ub( const char* file, unsigned char** data, 
               unsigned int *w,unsigned int *h)
{
    unsigned char *idata = 0;
    unsigned int channels;
    
    if (loadPPM( file, &idata, w, h, &channels)) {
        // pad 4th component
        int size = *w * *h;
        *data = new unsigned char[size * 4];
        unsigned char *ptr = *data;
        for(int i=0; i<size; i++) {
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = 0;
        }
        return CUTTrue;
    }
    else
    {
        return CUTFalse;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Load PGM image file (with unsigned int as data element type)
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutLoadPGMi( const char* file, unsigned int** data, 
             unsigned int *w, unsigned int *h) 
{
    return loadPGMt( file, data, w, h);
}

////////////////////////////////////////////////////////////////////////////////
//! Load PGM image file (with unsigned short as data element type)
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutLoadPGMs( const char* file, unsigned short** data, 
             unsigned int *w, unsigned int *h) 
{
    return loadPGMt( file, data, w, h);
}

////////////////////////////////////////////////////////////////////////////////
//! Load PGM image file (with float as data element type)
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutLoadPGMf( const char* file, float** data, 
             unsigned int *w, unsigned int *h) 
{
    return loadPGMt( file, data, w, h);
}

////////////////////////////////////////////////////////////////////////////////
//! Save PGM image file (with unsigned char as data element type)
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutSavePGMub( const char* file, unsigned char *data, 
              unsigned int w, unsigned int h) 
{
    return savePPM( file, data, w, h, 1);
}

////////////////////////////////////////////////////////////////////////////////
//! Save PPM image file (with unsigned char as data element type)
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutSavePPMub( const char* file, unsigned char *data, 
              unsigned int w, unsigned int h) 
{
    return savePPM( file, data, w, h, 3);
}

////////////////////////////////////////////////////////////////////////////////
//! Save PPM image file (with unsigned char as data element type, padded to 4 byte)
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutSavePPM4ub( const char* file, unsigned char *data, 
               unsigned int w, unsigned int h) 
{
    // strip 4th component
    int size = w * h;
    unsigned char *ndata = new unsigned char[size*3];
    unsigned char *ptr = ndata;
    for(int i=0; i<size; i++) {
        *ptr++ = *data++;
        *ptr++ = *data++;
        *ptr++ = *data++;
        data++;
    }
    
    return savePPM( file, ndata, w, h, 3);
}

////////////////////////////////////////////////////////////////////////////////
//! Save PGM image file (with unsigned int as data element type)
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutSavePGMi( const char* file, unsigned int *data, 
             unsigned int w, unsigned int h) 
{
    return savePGMt( file, data, w, h);
}

////////////////////////////////////////////////////////////////////////////////
//! Save PGM image file (with unsigned short  as data element type)
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutSavePGMs( const char* file, unsigned short *data, 
             unsigned int w, unsigned int h) 
{
    return savePGMt( file, data, w, h);
}

////////////////////////////////////////////////////////////////////////////////
//! Save PGM image file (with unsigned int as data element type)
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutSavePGMf( const char* file, float *data, 
             unsigned int w, unsigned int h) 
{
    return savePGMt( file, data, w, h);
}

////////////////////////////////////////////////////////////////////////////////
//! Check if command line argument \a flag-name is given
//! @return 1 if command line argument \a flag_name has been given, otherwise 0
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param flag_name  name of command line flag
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutCheckCmdLineFlag( const int argc, const char** argv, const char* flag_name) 
{
    CUTBoolean ret_val = CUTFalse;

    try 
    {
        // initalize 
        CmdArgReader::init( argc, argv);

        // check if the command line argument exists
        if( CmdArgReader::existArg( flag_name)) 
        {
            ret_val = CUTTrue;
        }
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string." << std::endl;
    } 

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type int
//! @return CUTTrue if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise CUTFalse
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutGetCmdLineArgumenti( const int argc, const char** argv, 
                        const char* arg_name, int* val) 
{
    CUTBoolean ret_val = CUTFalse;

    try 
    {
        // initialize
        CmdArgReader::init( argc, argv);

        // access argument
        const int* v = CmdArgReader::getArg<int>( arg_name);
        if( NULL != v) 
        {
            // assign value
            *val = *v;
            ret_val = CUTTrue;
        }
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string." << std::endl;
    } 

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type float
//! @return CUTTrue if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise CUTFalse
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutGetCmdLineArgumentf( const int argc, const char** argv, 
                       const char* arg_name, float* val) 
{
    CUTBoolean ret_val = CUTFalse;

    try 
    {
        // initialize
        CmdArgReader::init( argc, argv);

        // access argument
        const float* v = CmdArgReader::getArg<float>( arg_name);
        if( NULL != v) 
        {
            // assign value
            *val = *v;
            ret_val = CUTTrue;
        }
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string." << std::endl;
    } 

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type string
//! @return CUTTrue if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise CUTFalse
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutGetCmdLineArgumentstr( const int argc, const char** argv, 
                         const char* arg_name, char** val) 
{
    CUTBoolean ret_val = CUTFalse;

    try 
    {
        // initialize
        CmdArgReader::init( argc, argv);

        // access argument
        const std::string* v = CmdArgReader::getArg<std::string>( arg_name);
        if( NULL != v) 
        {

            // allocate memory for the string
            *val = new char[v->length() + 1];
            // copy from string to c_str
            strcpy( *val, v->c_str());
            ret_val = CUTTrue;
        }
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string."<< std::endl;
    } 

    return ret_val;

}
////////////////////////////////////////////////////////////////////////////////
//! Check for OpenGL error
//! @return CUTTrue if no GL error has been encountered, otherwise 0
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
//! @note The GL error is listed on stderr
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutCheckErrorGL( const char* file, const int line) 
{
    CUTBoolean ret_val = CUTTrue;

    try 
    {
        // check for error
        ErrorChecker::checkErrorGL( file, line);
    }
    catch( const std::exception& ex) 
    {
        // print where the exception occured
        std::cerr << ex.what() << std::endl;
        ret_val = CUTFalse;
    }

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Extended assert
//! @return CUTTrue if the condition \a val holds, otherwise CUTFalse
//! @param val  condition to test
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
////////////////////////////////////////////////////////////////////////////////
CUTBoolean
cutCheckCondition( int val, const char* file, const int line) 
{
    CUTBoolean ret_val = CUTTrue;

    try 
    {
        // check for error
        ErrorChecker::condition( (0 == val) ? false : true, file, line);
    }
    catch( const std::exception& ex) 
    {
        // print where the exception occured
        std::cerr << ex.what() << std::endl;
        ret_val = CUTFalse;
    }

    return ret_val;
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutComparef( const float* reference, const float* data,
            const unsigned int len ) 
{
    const float epsilon = 0.0;
    return compareData( reference, data, len, epsilon);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two integer arrays
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
CUTBoolean  
cutComparei( const int* reference, const int* data,
            const unsigned int len ) 
{
    const int epsilon = 0;
    return compareData( reference, data, len, epsilon);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two integer arrays
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
CUTBoolean  
cutCompareub( const unsigned char* reference, const unsigned char* data,
             const unsigned int len ) 
{
    const int epsilon = 0;
    return compareData( reference, data, len, epsilon);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays with an epsilon tolerance for equality
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutComparefe( const float* reference, const float* data,
             const unsigned int len, const float epsilon ) 
{
    return compareData( reference, data, len, epsilon);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays using L2-norm with an epsilon tolerance for equality
//! @return  CUTTrue if \a reference and \a data are identical, 
//!          otherwise CUTFalse
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutCompareL2fe( const float* reference, const float* data,
                const unsigned int len, const float epsilon ) 
{
    CUT_CONDITION( epsilon >= 0);

    float error = 0;
    float ref = 0;

    for( unsigned int i = 0; i < len; ++i) {

        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);
    if (fabs(ref) < 1e-7) {
#ifdef _DEBUG
        std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
        return CUTFalse;
    }
    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
#ifdef _DEBUG
    if( ! result) 
    {
        std::cerr << "ERROR, l2-norm error " 
            << error << " is greater than epsilon " << epsilon << "\n";
    }
#endif

    return result ? CUTTrue : CUTFalse;
}

////////////////////////////////////////////////////////////////////////////////
//! Timer functionality

////////////////////////////////////////////////////////////////////////////////
//! Create a new timer
//! @return CUTTrue if a time has been created, otherwise false
//! @param  name of the new timer, 0 if the creation failed
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutCreateTimer( unsigned int* name) 
{
    *name = StopWatch::create();

    return (0 != name) ? CUTTrue : CUTFalse;
}


////////////////////////////////////////////////////////////////////////////////
//! Delete a timer
//! @return CUTTrue if a time has been deleted, otherwise false
//! @param  name of the timer to delete
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutDeleteTimer( unsigned int name) 
{
    CUTBoolean retval = CUTTrue;

    try 
    {
        StopWatch::destroy( name);
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
        retval = CUTFalse;
    }

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Start the time with name \a name
//! @param name  name of the timer to start
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutStartTimer( const unsigned int name) 
{
    CUTBoolean retval = CUTTrue;

#ifdef _DEBUG
    try 
    {
        StopWatch::get( name).start();
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
        retval = CUTFalse;
    }
#else
    StopWatch::get( name).start();
#endif

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop the time with name \a name. Does not reset.
//! @param name  name of the timer to stop
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutStopTimer( const unsigned int name) 
{
    CUTBoolean retval = CUTTrue;

#ifdef _DEBUG
    try 
    {
        StopWatch::get( name).stop();
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
        retval = CUTFalse;
    }
#else
    StopWatch::get( name).stop();
#endif

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Resets the timer's counter.
//! @param name  name of the timer to reset.
////////////////////////////////////////////////////////////////////////////////
CUTBoolean 
cutResetTimer( const unsigned int name)
{
    CUTBoolean retval = CUTTrue;

#ifdef _DEBUG
    try 
    {
        StopWatch::get( name).reset();
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
        retval = CUTFalse;
    }
#else
    StopWatch::get( name).reset();
#endif

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Return the average time for timer execution as the total time
//! for the timer dividied by the number of completed (stopped) runs the timer 
//! has made.
//! Excludes the current running time if the timer is currently running.
//! @param name  name of the timer to return the time of
////////////////////////////////////////////////////////////////////////////////
float 
cutGetAverageTimerValue( const unsigned int name)
{
    float time = 0.0;

#ifdef _DEBUG
    try 
    {
        time = StopWatch::get( name).getAverageTime();
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
    }
#else
    time = StopWatch::get( name).getAverageTime();
#endif

    return time;
}

////////////////////////////////////////////////////////////////////////////////
//! Total execution time for the timer over all runs since the last reset
//! or timer creation.
//! @param name  name of the timer to obtain the value of.
////////////////////////////////////////////////////////////////////////////////
float 
cutGetTimerValue( const unsigned int name) 
{  
    float time = 0.0;

#ifdef _DEBUG
    try 
    {
        time = StopWatch::get( name).getTime();
    }
    catch( const std::exception& ex) 
    {
        std::cerr << "WARNING: " << ex.what() << std::endl;
    }
#else
    time = StopWatch::get( name).getTime();
#endif

    return time;
}



////////////////////////////////////////////////////////////////////////////////
//! TBD
////////////////////////////////////////////////////////////////////////////////
void
cutCheckBankAccess( unsigned int tidx, unsigned int tidy, unsigned int tidz,
                   unsigned int bdimx, unsigned int bdimy, unsigned int bdimz,
                   const char* file, const int line, const char* aname,
                   const int index) 
{
    BankChecker::getHandle()->access( tidx, tidy, tidz, 
                                      bdimx, bdimy, bdimz,
                                      file, line, aname, index );
}

