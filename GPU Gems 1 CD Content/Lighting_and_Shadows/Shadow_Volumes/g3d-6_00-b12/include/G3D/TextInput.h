/**
 @file TextInput.h

 Simple text lexer/tokenizer.

 @maintainer Morgan McGuire, morgan@graphics3d.com

 @cite Based on a lexer written by Aaron Orenstein. 

 @created 2002-11-27
 @edited  2003-10-09
  Copyright 2000-2004, Morgan McGuire.
  All rights reserved.
 */

#ifndef G3D_TEXTINPUT_H
#define G3D_TEXTINPUT_H

#include "G3D/Array.h"
#include <string>
#include <queue>
#include <ctype.h>
#include <stdio.h>

namespace G3D {

/**
 For use with TextInput.
 */
class Token {
public:
    /**
     Strings are enclosed in quotes, symbols are not.
     */
    enum Type {STRING, SYMBOL, NUMBER, END};

private:
    friend class TextInput;

    std::string             _string;
    int                     _line;
    int                     _character;
    Type                    _type;

public:


    Token() : _string(""), _line(0), _character(0), _type(END) {}

    Token(Type t, const std::string& s, int L, int c) : _string(s), _line(L), _character(c), _type(t) {}


    Type type() const {
        return _type;
    }

    std::string string() const {
        return _string;
    }

    int line() const {
        return _line;
    }

    int character() const {
        return _character;
    }

    /** Return the numer value */
    double number() const {
        if (_type == NUMBER) {
            double n;
            if ((_string.length() > 2) &&
                (_string[0] == '0') &&
                (_string[1] == 'x')) {
                // Hex
                uint32 i;
                sscanf(_string.c_str(), "%x", &i);
                n = i;
            } else {
                sscanf(_string.c_str(), "%lg", &n);
            }
            return n;
        } else {
            return 0;
        }
    }


};


/**
 A simple tokenizer for reading text files.  TextInput handles C++ like
 text including single line comments, block comments, quoted strings with
 escape sequences, and operators.  

 The special ".." and "..." tokens are recognized in addition to normal C++ operators.

 Negative numbers are handled specially-- see the note on read().

  e.g.
  <pre>
  TextInput i(TextInput::FROM_STRING, "name = \"Max\", height = 6");

  Token d;

  t = t.read(); 
  debugAssert(t.type == Token::SYMBOL);
  debugAssert(t.sval == "name");

  t.read();
  debugAssert(t.type == Token::SYMBOL);
  debugAssert(t.sval == "=");

  std::string name = t.read().sval;
  t.read();
  </pre>

  There is no TextOutput class because printf and character streams fill
  that role nicely in C++.

 */
class TextInput {
public:

    class Options {
    public:
        /** If true, single line comments beginning with // are ignored.
            Default is true. */
        bool                cppComments;

        /** If true, "-1" parses as the number -1 instead of the symbol "-" followed
            by the number 1.  Default is true.*/
        bool                signedNumbers;

        Options () : cppComments(true), signedNumbers(true) {}
    };

private:

    std::deque<Token>       stack;

    /**
     The character you'll get if you peek 1 ahead
     */
    Array<char>             peekChar;

    /**
     Characters to be parsed.
     */
    Array<char>             buffer;

    /**
     Last character index consumed.
     */
    int                     bufferLast;
    int                     lineNumber;

    /**
     Number of characters from the beginning of the line. 
     */
    int                     charNumber;
    std::string             sourceFile;
    
    Options                 options;

    void init() {
        sourceFile = "";
        charNumber = 0;
        bufferLast = -1;
        lineNumber = 1;
    }

    /**
     Returns the next character and sets filename and linenumber
     to reflect the location where the character came from.
     */
    int popNextChar();

    inline char peekNextChar() {
        return buffer[bufferLast + 1];

    }

    inline void pushNextChar(char c) {
        if (c != EOF) {
            debugAssert(c == buffer[bufferLast]);
            bufferLast--;
        }
    }

    /** Read the next token or EOF */
    Token nextToken();

public:

    class TokenException {
    public:
        std::string     sourceFile;
        int             line;
        int             character;

        /** Pre-formatted error message */
        std::string     message;

        virtual ~TokenException() {}
    protected:

        TokenException(
            const std::string&  src,
            int                 ln,
            int                 ch);

    };

    /** Thrown by the read methods. */
    class WrongTokenType : public TokenException {
    public:
        Token::Type     expected;
        Token::Type     actual;

        WrongTokenType(
            const std::string&  src,
            int                 ln,
            int                 ch,
            Token::Type         e,
            Token::Type         a);
    };

    class WrongSymbol : public TokenException {
    public:
        std::string             expected;
        std::string             actual;

        WrongSymbol(
            const std::string&  src,
            int                 ln,
            int                 ch,
            const std::string&  e,
            const std::string&  a);
    };

    TextInput(const std::string& filename, const Options& options = Options());

    enum FS {FROM_STRING};
    /** Creates input directly from a string.
        The first argument must be TextInput::FROM_STRING.*/
    TextInput(FS fs, const std::string& str, const Options& options = Options());

    /** Returns true while there are tokens remaining. */
    bool hasMore();

    /** Read the next token (which will be the END token if ! hasMore()).
    
        Signed numbers can be handled in one of two modes.  If the option 
        TextInput::Options::signedNumbers is true,
        A '+' or '-' immediately before a number is prepended onto that number and
        if there is intervening whitespace, it is read as a separate symbol.

        If TextInput::Options::signedNumbers is false,
        read() does not distinguish between a plus or minus symbol next
        to a number and a positive/negative number itself.  For example, "x - 1" and "x -1"
        will be parsed the same way by read().  
        
        In both cases, readNumber() will contract a leading "-" or "+" onto
        a number.
    */
    Token read();

    /** Throws WrongTokenType if the next token is not a number or
        a plus or minus sign followed by a number.  In the latter case,
        this will read two tokens and combine them into a single number. 
        When an exception is thrown no tokens are consumed.*/
    double readNumber();

    /** Reads a string or throws WrongTokenType.  The quotes are taken off of strings. */
    std::string readString();
    
    /** Reads a symbol or throws WrongTokenType */
    std::string readSymbol();

    /** Reads a specific symbol or throws either WrongTokenType or WrongSymbol*/
    void readSymbol(const std::string& symbol);

    /** Look at the next token but don't extract it */
    Token peek();

    /** Take a previously read token and push it back (used
        in the rare case where more than one token of read-ahead
        is needed so peek doesn't suffice). */
    void push(const Token& t);
};

} // namespace

#endif
