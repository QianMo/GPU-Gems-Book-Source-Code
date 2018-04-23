/*********************************************************************NVMH3****
Path:  E:\nvidia\devrel\NVSDK\Common\include\nv_core
File:  nv_attribute.h

Copyright NVIDIA Corporation 2002
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.


Comments:




******************************************************************************/

#ifndef _nv_attribute_h_
#define _nv_attribute_h_

#ifdef WIN32
#pragma warning (disable:4251) 
#endif


#include <nv_nvb/nv_nvbdecl.h>
#include <nv_nvb/nv_streams.h>


// ----------------------------------------------------------------------------
// class nv_attribute
//
class DECLSPEC_NV_NVB nv_attribute
{
    typedef std::map<std::string,nv_attribute*> _dico;
    typedef _dico::value_type                   _dico_pair;
    typedef _dico::iterator                     _dico_it;
    typedef _dico::const_iterator               _dico_cit;
public:

    enum
    {
        NV_FLOAT                = 0x00000001,
        NV_CHAR                 = 0x00000002,
        NV_UNSIGNED_CHAR        = 0x00000004,
        NV_UNSIGNED_INT         = 0x00000008,
        NV_INT                  = 0x00000010,
        NV_SHORT                = 0x00000040,
        
        NV_ARRAY                = 0x00010000,
        NV_FLOAT_ARRAY          = 0x00010001,
        NV_CHAR_ARRAY           = 0x00010002,
        NV_UNSIGNED_CHAR_ARRAY  = 0x00010004,
        NV_UNSIGNED_INT_ARRAY   = 0x00010008,
        NV_INT_ARRAY            = 0x00010010,
        NV_ATTRIBUTE_ARRAY      = 0x00010020,
        NV_SHORT_ARRAY          = 0x00010040,    
        NV_STRING               = 0x00010002,
        
        NV_UNASSIGNED           = 0xFFFFFFFF
    };

    nv_attribute();
    nv_attribute(const nv_attribute & attr);
    ~nv_attribute();

    // l-value assignation
    nv_attribute &          operator=           (const char * str);
    nv_attribute &          operator=           (const float & val);
    nv_attribute &          operator=           (const unsigned int & val);
    nv_attribute &          operator=           (const int & val);
    nv_attribute &          operator=           (const unsigned char & val);
    nv_attribute &          operator=           (const char & val);
    nv_attribute &          operator=           (const short & val);
    nv_attribute &          operator=           (const nv_attribute & attr);

    // validation tests
    bool                    is_null             () const;
    bool                    is_valid            () const;

    // accessor interface
    nv_attribute &          operator[]          (const char * name);

    // array assignation
    void                    array               (const nv_attribute * attribs, unsigned int size);
    void                    array               (const float * fvector, unsigned int size);
    void                    array               (const char * str, unsigned int size);
    void                    array               (const unsigned char * str, unsigned int size);
    void                    array               (const unsigned int * array, unsigned int size);
    void                    array               (const int * array, unsigned int size);
    void                    array               (const short * array, unsigned int size);
    
    // r-value accessors...
    float                   as_float            () const;
    unsigned int            as_unsigned_int     () const;
    unsigned int            as_int              () const;
    short                   as_word             () const;
    unsigned char           as_uchar            () const;
    char                    as_char             () const;
    const char *            as_char_array       () const;
    const unsigned char *   as_unsigned_char_array() const;
    const int *             as_int_array        () const;
    const unsigned int *    as_unsigned_int_array() const;
    const float *           as_float_array      () const;
    const short *           as_short_array      () const;
    const char *            as_string           () const;
    const nv_attribute *    as_attribute_array  () const;

    // retrieve the size array of the array
    const unsigned int      get_size            () const;

    // retrieve the type of the attribute
    const unsigned int      get_type            () const;

    unsigned int            get_num_attributes  () const;
    const char *            get_attribute_name  (unsigned int i) const;
    const nv_attribute *    get_attribute       (unsigned int i) const;

    friend DECLSPEC_NV_NVB std::ostream & operator << (std::ostream & os, const nv_attribute & attr);
    friend nv_input_stream  & operator >> (nv_input_stream  & rInputStream,        nv_attribute & rAttribute);
    friend nv_output_stream & operator << (nv_output_stream & rOutputStream, const nv_attribute & oAttribute);

protected:
    // delete the container and reset the value to zero
    void                    erase               ();
    void                    copy_from           (const nv_attribute * attr);

private: 
    // dictionnary
    _dico                   _attr;

    // 32 bit union for data storage
    union
    {
        void *              _pdata;
        unsigned char       _ubyte;
        unsigned char       _byte;
        short               _word;
        unsigned int        _udword;
        unsigned int        _sdword;
        float               _float;
    };
    
    // to be only used for arrays only. The size is always in reference to the type.
    // example: if the type is float and size is 4, there is 3 floats in the array
    unsigned int    _size; 

    // used to tell whether an attribute contains a valid value as well as its type
    unsigned int    _flag;
};

#endif // _nv_attribute_h_
