template <unsigned int sizeA, unsigned int sizeB> class LUB {
public:
    enum SIZ{size=((sizeA==0&&sizeB==0)?1:
		   (sizeA==1)?sizeB:(sizeB==1)?sizeA:
		   (sizeA>sizeB)?sizeB:sizeA)
    };
};
template<> class LUB<1,1> {public:
    enum SIZ{size=1};
};
template<> class LUB<1,2> {public:
    enum SIZ{size=2};
};
template<> class LUB<2,1> {public:
    enum SIZ{size=2};
};
template<> class LUB<1,3> {public:
    enum SIZ{size=3};
};
template<> class LUB<3,1> {public:
    enum SIZ{size=3};
};
template<> class LUB<1,4> {public:
    enum SIZ{size=4};
};
template<> class LUB<4,1> {public:
    enum SIZ{size=4};
};
template<> class LUB<2,2> {public:
    enum SIZ{size=2};
};
template<> class LUB<2,3> {public:
    enum SIZ{size=2};
};
template<> class LUB<3,2> {public:
    enum SIZ{size=2};
};
template<> class LUB<2,4> {public:
    enum SIZ{size=2};
};
template<> class LUB<4,2> {public:
    enum SIZ{size=2};
};
template<> class LUB<3,3> {public:
    enum SIZ{size=3};
};
template<> class LUB<3,4> {public:
    enum SIZ{size=3};
};
template<> class LUB<4,3> {public:
    enum SIZ{size=3};
};
template<> class LUB<4,4> {public:
    enum SIZ{size=4};
};

template <class VALUE, unsigned int tsize> class vec;

template <class A, class B> class LCM {
public:
    typedef vec<typename LCM<typename A::TYPE , typename B::TYPE>::type,
                LUB<A::size,B::size>::size> type;
};
template <> class LCM<float,float> {public:
    typedef float type;
};
template <> class LCM<float,double> {public:
    typedef double type;
};
template <> class LCM<double,float> {public:
    typedef double type;
};
template <> class LCM<double,double> {public:
    typedef double type;
};
template <> class LCM<char,float> {public:
    typedef float type;
};
template <> class LCM<float,char> {public:
    typedef float type;
};
template <> class LCM<char,char> {public:
    typedef char type;
};
template <> class LCM<double,char> {public:
    typedef double type;
};
template <> class LCM<char,double> {public:
    typedef double type;
};


template <> class LCM<vec<float,1>,float> {public:
    typedef float type;
};
template <> class LCM<float,vec<float,1> > {public:
    typedef float type;
};
template <> class LCM<vec<float,1>,double> {public:
    typedef double type;
};
template <> class LCM<float,vec<double,1> > {public:
    typedef double type;
};
template <> class LCM<double,vec<float,1> > {public:
    typedef double type;
};
template <> class LCM<vec<double,1>,float > {public:
    typedef double type;
};
template <> class LCM<vec<double,1>,double> {public:
    typedef double type;
};
template <> class LCM<double,vec<double,1> > {public:
    typedef double type;
};

template <> class LCM<char,vec<float,1> > {public:
    typedef float type;
};
template <> class LCM<vec<char,1>,float> {public:
    typedef float type;
};
template <> class LCM<float,vec<char,1> > {public:
    typedef float type;
};
template <> class LCM<vec<float,1>,char> {public:
    typedef float type;
};
template <> class LCM<char,vec<char,1> > {public:
    typedef char type;
};
template <> class LCM<vec<char,1>,char> {public:
    typedef char type;
};
template <> class LCM<double,vec<char,1> > {public:
    typedef double type;
};
template <> class LCM<vec<double,1>,char> {public:
    typedef double type;
};
template <> class LCM<vec<char,1>,double> {public:
    typedef double type;
};
template <> class LCM<char,vec<double,1> > {public:
    typedef double type;
};

template <> class LCM<vec<float,2>,float> {public:
    typedef float type;
};
template <> class LCM<float,vec<float,2> > {public:
    typedef float type;
};
template <> class LCM<vec<float,2>,double> {public:
    typedef double type;
};
template <> class LCM<float,vec<double,2> > {public:
    typedef double type;
};
template <> class LCM<double,vec<float,2> > {public:
    typedef double type;
};
template <> class LCM<vec<double,2>,float > {public:
    typedef double type;
};
template <> class LCM<vec<double,2>,double> {public:
    typedef double type;
};
template <> class LCM<double,vec<double,2> > {public:
    typedef double type;
};

template <> class LCM<char,vec<float,2> > {public:
    typedef float type;
};
template <> class LCM<vec<char,2>,float> {public:
    typedef float type;
};
template <> class LCM<float,vec<char,2> > {public:
    typedef float type;
};
template <> class LCM<vec<float,2>,char> {public:
    typedef float type;
};
template <> class LCM<char,vec<char,2> > {public:
    typedef char type;
};
template <> class LCM<vec<char,2>,char> {public:
    typedef char type;
};
template <> class LCM<double,vec<char,2> > {public:
    typedef double type;
};
template <> class LCM<vec<double,2>,char> {public:
    typedef double type;
};
template <> class LCM<vec<char,2>,double> {public:
    typedef double type;
};
template <> class LCM<char,vec<double,2> > {public:
    typedef double type;
};
template <> class LCM<vec<float,3>,float> {public:
    typedef float type;
};
template <> class LCM<float,vec<float,3> > {public:
    typedef float type;
};
template <> class LCM<vec<float,3>,double> {public:
    typedef double type;
};
template <> class LCM<float,vec<double,3> > {public:
    typedef double type;
};
template <> class LCM<double,vec<float,3> > {public:
    typedef double type;
};
template <> class LCM<vec<double,3>,float > {public:
    typedef double type;
};
template <> class LCM<vec<double,3>,double> {public:
    typedef double type;
};
template <> class LCM<double,vec<double,3> > {public:
    typedef double type;
};

template <> class LCM<char,vec<float,3> > {public:
    typedef float type;
};
template <> class LCM<vec<char,3>,float> {public:
    typedef float type;
};
template <> class LCM<float,vec<char,3> > {public:
    typedef float type;
};
template <> class LCM<vec<float,3>,char> {public:
    typedef float type;
};
template <> class LCM<char,vec<char,3> > {public:
    typedef char type;
};
template <> class LCM<vec<char,3>,char> {public:
    typedef char type;
};
template <> class LCM<double,vec<char,3> > {public:
    typedef double type;
};
template <> class LCM<vec<double,3>,char> {public:
    typedef double type;
};
template <> class LCM<vec<char,3>,double> {public:
    typedef double type;
};
template <> class LCM<char,vec<double,3> > {public:
    typedef double type;
};


template <> class LCM<vec<float,4>,float> {public:
    typedef float type;
};
template <> class LCM<float,vec<float,4> > {public:
    typedef float type;
};
template <> class LCM<vec<float,4>,double> {public:
    typedef double type;
};
template <> class LCM<float,vec<double,4> > {public:
    typedef double type;
};
template <> class LCM<double,vec<float,4> > {public:
    typedef double type;
};
template <> class LCM<vec<double,4>,float > {public:
    typedef double type;
};
template <> class LCM<vec<double,4>,double> {public:
    typedef double type;
};
template <> class LCM<double,vec<double,4> > {public:
    typedef double type;
};

template <> class LCM<char,vec<float,4> > {public:
    typedef float type;
};
template <> class LCM<vec<char,4>,float> {public:
    typedef float type;
};
template <> class LCM<float,vec<char,4> > {public:
    typedef float type;
};
template <> class LCM<vec<float,4>,char> {public:
    typedef float type;
};
template <> class LCM<char,vec<char,4> > {public:
    typedef char type;
};
template <> class LCM<vec<char,4>,char> {public:
    typedef char type;
};
template <> class LCM<double,vec<char,4> > {public:
    typedef double type;
};
template <> class LCM<vec<double,4>,char> {public:
    typedef double type;
};
template <> class LCM<vec<char,4>,double> {public:
    typedef double type;
};
template <> class LCM<char,vec<double,4> > {public:
    typedef double type;
};






template <class A, class B> class COMMON_CHAR {
public:
   typedef vec<typename COMMON_CHAR<typename A::TYPE , 
                                    typename B::TYPE>::type,
               LUB<A::size,B::size>::size> type;
};
template <> class COMMON_CHAR<float,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,double> {public:
    typedef char type;
};



template <> class COMMON_CHAR<vec<float,1>,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<float,1> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<float,1>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<double,1> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<float,1> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,1>,float > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,1>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<double,1> > {public:
    typedef char type;
};

template <> class COMMON_CHAR<char,vec<float,1> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,1>,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<char,1> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<float,1>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,vec<char,1> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,1>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<char,1> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,1>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,1>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,vec<double,1> > {public:
    typedef char type;
};


template <> class COMMON_CHAR<vec<float,2>,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<float,2> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<float,2>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<double,2> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<float,2> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,2>,float > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,2>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<double,2> > {public:
    typedef char type;
};

template <> class COMMON_CHAR<char,vec<float,2> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,2>,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<char,2> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<float,2>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,vec<char,2> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,2>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<char,2> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,2>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,2>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,vec<double,2> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<float,3>,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<float,3> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<float,3>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<double,3> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<float,3> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,3>,float > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,3>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<double,3> > {public:
    typedef char type;
};

template <> class COMMON_CHAR<char,vec<float,3> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,3>,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<char,3> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<float,3>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,vec<char,3> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,3>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<char,3> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,3>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,3>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,vec<double,3> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<float,4>,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<float,4> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<float,4>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<double,4> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<float,4> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,4>,float > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,4>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<double,4> > {public:
    typedef char type;
};

template <> class COMMON_CHAR<char,vec<float,4> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,4>,float> {public:
    typedef char type;
};
template <> class COMMON_CHAR<float,vec<char,4> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<float,4>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,vec<char,4> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,4>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<double,vec<char,4> > {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<double,4>,char> {public:
    typedef char type;
};
template <> class COMMON_CHAR<vec<char,4>,double> {public:
    typedef char type;
};
template <> class COMMON_CHAR<char,vec<double,4> > {public:
    typedef char type;
};
