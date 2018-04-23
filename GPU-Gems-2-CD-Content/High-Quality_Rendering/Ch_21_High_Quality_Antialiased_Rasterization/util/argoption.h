/////////////////////////////////////////////////////////////////////////////
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// (This is the Modified BSD License)
/////////////////////////////////////////////////////////////////////////////


#ifndef ARGOPTION_H
#define ARGOPTION_H

class ArgOption {

 public:

    ArgOption (const char *str);
    ~ArgOption();
    
    int initialize ();
    
    int parameter_count () const { return count; }
    const char *name() const { return flag; }

    bool is_flag() const { return type == Flag; }
    bool is_sublist() const { return type == Sublist; }
    bool is_regular() const { return type == Regular; }
    
    void add_parameter (int i, void *p);
    void set_parameter (int i, char *argv);

    void add_argument (char *argv);
    int invoke_callback() const;

    void found_on_command_line() { repetitions++; }
    int parsed_count() { return repetitions; }
    
 private:
    
    enum OptionType { None, Regular, Flag, Sublist };

    const char *format;                         // original format string
    char *flag;                                 // just the -flag_foo part
    char *code;                                 // paramter types, eg "df"
    OptionType type;                    
    int count;                                  // number of parameters
    void **param;                               // pointers to app data vars
    int (*callback) (int argc, char **argv);
    int repetitions;                            // number of times on cmd line
    int argc;                                   // for sublist storage
    char **argv;
};


#endif // ARGOPTION_H
