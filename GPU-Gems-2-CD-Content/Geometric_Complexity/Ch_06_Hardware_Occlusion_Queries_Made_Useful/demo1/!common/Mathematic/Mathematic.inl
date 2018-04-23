//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
namespace Math {

template<class A>
inline void clamp(A& value, const A& min, const A& max) {
	if(value < min) {
		value = min;
	}
	else {
		if(value > max) {
			value = max;
		}
	}
}

template<class A>
inline A clamp(const A& value, const A& min, const A& max) {
	if(value < min) {
		return min;
	}
	else {
		if(value > max) {
			return max;
		}
	}
	return value;
}

template<class T> bool between(const T& a, const T& b, const T& value) {
	if(a < b) {
		return (a < value) && (value < b);
	}
	else {
		return (b < value) && (value < a);
	}
}

// return a random number form the interval [start .. stop]
template<class T> T randRange(const T& from, const T& to) {
	if(from == to) 
		return from;
	const T range = 1+to-from;
	return to + (range*rand())/RAND_MAX;
}

template<class REAL> REAL relativeEpsilon(const REAL& a, const REAL& epsilon) {
	REAL relEpsilon;
	maximum(relEpsilon,abs(a*epsilon),epsilon);
	return relEpsilon;
}

template<class REAL> bool alike(const REAL& a, const REAL& b, const REAL& epsilon) {
	if(a == b) {
		return true;
	}
	REAL relEps(relativeEpsilon(a,epsilon));
	return (a-relEps <= b) && (b <= a+relEps);
}

/*	| a1 a2 |
	| b1 b2 | calculate the determinent of a 2x2 matrix in the from*/
template<class REAL> inline REAL det2x2(const REAL& a1, const REAL& a2, 
								    const REAL& b1, const REAL& b2) {
	return a1*b2 - b1*a2;
}

/*	| a1 a2 a3 |
	| b1 b2 b3 |
	| c1 c2 c3 | calculate the determinent of a 3x3 matrix*/
template<class REAL> inline REAL det3x3(const REAL& a1, const REAL& a2, const REAL& a3,
									const REAL& b1, const REAL& b2, const REAL& b3,
									const REAL& c1, const REAL& c2, const REAL& c3) {
	return a1*det2x2(b2,b3,c2,c3) - b1*det2x2(a2,a3,c2,c3) +
			c1*det2x2(a2,a3,b2,b3);
}

template<class REAL> inline REAL deg2_n180_180(const REAL& vDeg) {
    REAL tmp;
    if (abs(vDeg) >= 360) {
        tmp = mod<REAL>(vDeg,360);
    }
    else {
        tmp = vDeg;
    }
    if(tmp > 180) {
        tmp -= 360;
    }
    else {
        if(tmp <= -180) {
            tmp += 360;
        }
    }
    return tmp;
}

template<class REAL>
inline void interpolate(const REAL& dest, const REAL& step, REAL& value) {
	if(dest != value) {
		REAL step2(abs(step));
		//determine direction
		if(value > dest) {
			step2 = -step2;
		}
		REAL newValue(value+step2);
		if(between(value,dest,newValue)) {
			value = newValue;
		}
		else {
			value = dest;
		}
	}
}

//namespace
}
