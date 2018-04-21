#ifndef ASSERT_CG_H
#define ASSERT_CG_H
// -------------------------------------------------------------------
// Contents:
//      CG_ASSERT_NO_ERROR macro
//          This macro allows for simple Cg error checking.
//
// Author:
//      Frank Jargstorff (2003)
// -------------------------------------------------------------------


        // cg_assert
        //
        // Description:
        //      Helper function handling the error reporting.
        //          The reason this function is not defined inline
        //      as part of the macro is to allow for setting a 
        //      breakpoint inside the function to facilitate debugging.
        //
void cg_assert(const char * zFile, unsigned int nLine);


        // CG_ASSERT_NO_ERROR
        //
        // Description:
        //      Tests for Cg error codes.
        //          If an Cg error occured this assertion will
        //      fail, print an error message, and terminate program
        //      exection.
        //          This macro should be redefined the empty 
        //      function for release builds.
        //
#define CG_ASSERT_NO_ERROR {cg_assert(__FILE__, __LINE__);}

#endif // ASSERT_CG_H