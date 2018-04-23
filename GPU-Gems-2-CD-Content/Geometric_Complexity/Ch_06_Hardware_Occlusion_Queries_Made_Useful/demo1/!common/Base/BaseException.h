//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef BaseExceptionH
#define BaseExceptionH
//---------------------------------------------------------------------------
#include <exception>
#include <string>
//---------------------------------------------------------------------------
class BaseException : public std::exception {
protected:
	std::string msg;
public:
	BaseException() { };
	BaseException(const std::string& vMsg) { msg = vMsg; };
	virtual const char *what() const throw() { return msg.c_str(); }
	const std::string getMsg() const throw() { return msg; }
};

struct ExistException : public BaseException {
	ExistException(const std::string& vMsg): BaseException(vMsg) { };
};

struct FileException : public BaseException {
	FileException(const std::string& vMsg): BaseException(vMsg) { };
};

struct SingeltonException : public BaseException {
	SingeltonException(): BaseException("tried to create Invalid Instance") { };
	SingeltonException(const std::string& vMsg): BaseException(vMsg) { };
};

#endif
