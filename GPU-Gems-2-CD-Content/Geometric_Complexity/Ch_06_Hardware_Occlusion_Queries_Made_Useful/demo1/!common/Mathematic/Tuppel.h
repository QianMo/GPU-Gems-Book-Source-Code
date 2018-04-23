//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef TuppelH
#define TuppelH
//---------------------------------------------------------------------------
namespace Math {

template<class REAL, const unsigned SIZE>
struct Tuppel {
	typedef REAL ElementType;
	typedef Tuppel<REAL,SIZE> T;
protected:
	REAL array[SIZE];
	void init(const REAL p[SIZE]) {
		for(unsigned i = 0; i < SIZE; i++)
			array[i] = p[i]; 
	}

public:
	// construction
	Tuppel() { /* no initialization -> performance */ }
	Tuppel(const Tuppel& v) { 
		init(v.array);
	}

	Tuppel(const REAL p[SIZE]) { 
		init(p);
	}

	static const unsigned size() { return SIZE; }

	REAL* addr() { return array; }
	const REAL* addr() const { return array; }
	const REAL operator[](const unsigned id) const { return array[id]; }
	REAL& operator[](const unsigned id) { return array[id]; }

	Tuppel& operator=(const Tuppel& v) {
		for(unsigned i = 0; i < SIZE; i++)
			array[i] = v[i]; 
		return *this;
	}

	bool operator==(const Tuppel& v) const {
		bool result = (array[0] == v[0]);
		for(unsigned i = 1; i < SIZE; i++)
			result &= (array[i] == v[i]);
		return result; 
	}

	bool operator!=(const Tuppel& v) const { return !(*this == v); }

	Tuppel operator-() const {
		Tuppel v;
		for(unsigned i = 0; i < SIZE; i++)
			v[i] = -array[i];
		return v; 
	}

	Tuppel& operator+=(const Tuppel& v) {
		for(unsigned i = 0; i < SIZE; i++)
			array[i] += v[i];
		return *this;
	}

	Tuppel& operator-=(const Tuppel& v) {
		for(unsigned i = 0; i < SIZE; i++)
			array[i] -= v[i];
		return *this;
	}

	Tuppel& operator*=(const REAL& r) {
		for(unsigned i = 0; i < SIZE; i++)
			array[i] *= r;
		return *this;
	}

	Tuppel& operator/=(const REAL& r) { return operator*=(REAL(1.0)/r); }

	Tuppel operator+(const Tuppel& v) const { return Tuppel(*this) += v; }
	Tuppel operator-(const Tuppel& v) const { return Tuppel(*this) -= v; }
	Tuppel operator*(const REAL& r) const { return Tuppel(*this) *= r; }
	Tuppel operator/(const REAL& r) const { return Tuppel(*this) /= r; }

	bool alike(const Tuppel& v, const REAL& epsilon) const {
		bool result = Math::alike(array[0],v[0],epsilon);
		for(unsigned i = 1; i < SIZE; i++)
			result &= Math::alike(array[i],v[i],epsilon);
		return result; 
	}

	REAL getMax() const {
		REAL m = array[0];
		for(unsigned i = 1; i < SIZE; i++) {
			if(m < array[i]) {
				m = array[i];
			}
		}
		return m;	
	}

	REAL getMin() const {
		REAL m = array[0];
		for(unsigned i = 1; i < SIZE; i++) {
			if(array[i] < m) {
				m = array[i];
			}
		}
		return m;	
	}
	
	REAL product() const {
		REAL p = array[0];
		for(unsigned i = 1; i < SIZE; i++) {
			p *= array[i];
		}
		return p;	
	}

	T abs() const {
		T t;
		for(unsigned i = 0; i < SIZE; i++) {
			t[i] = Math::abs(array[i]);
		}
		return t;
	}

	template<class T> Tuppel<T,SIZE> convert2() const {
		Tuppel<T,SIZE> m;
		for(unsigned i = 0; i < SIZE; i++) {
			m[i] = T(array[i]);
		}
		return m;
	}

};
template<class REAL, unsigned SIZE> 
Tuppel<REAL,SIZE> operator*(const REAL& r, const Tuppel<REAL,SIZE>& v) {
	Tuppel<REAL,SIZE> res(v);
	res *= r;
	return res;
}


//namespace
};
#endif