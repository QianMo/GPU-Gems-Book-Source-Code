#ifndef _TEXT_INC
#define _TEXT_INC

/*
A classic, used everywhere to read formatted text files without using Lex
and Yacc. Written in C around 1995
Re-vamped a million times, and included in several programs
Converted to C++ in 1.7.98 to use it in the Rayman project.
*/

class text
        {
        char *data;
        unsigned int sl;
        unsigned int size;
        public:
				text();
				text(char*);

                void create(char *);
                char *getword();
				char *getcommaword();
                int getint();
                double getfloat();

                int countword(char *);
				int countwordfromhere(char *);
                int countchar(char);
                void reset();
                void destroy();
                void goback();
				void seek(char *);
                int eof();
				~text();
        };

#endif









