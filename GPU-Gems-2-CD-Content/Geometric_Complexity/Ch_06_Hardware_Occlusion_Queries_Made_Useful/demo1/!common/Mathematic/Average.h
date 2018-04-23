//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef AverageH
#define AverageH
//---------------------------------------------------------------------------
#include <list>
//---------------------------------------------------------------------------
namespace Math {

template<class T>
class Average {
	typedef std::list<T> DATA;
	DATA data;
	unsigned size;
public:
	Average(const unsigned count): size(count) { }
	
	void add(const T& t) {
		if(data.size() > size) {
			data.erase(data.begin());
		}
		data.push_back(t);
	}

	void setSampleCount(const unsigned count) { 
		size = count; 
		while(data.size() > size) {
			data.erase(data.begin());
		}
	}

	T avg() const {
		T result = data.front();
		DATA::const_iterator i = data.begin();
		i++;
		for(; i != data.end(); i++) {
			result += *i;
		}
		result /= data.size();
		return result;
	}
};

//namespace
}
#endif
