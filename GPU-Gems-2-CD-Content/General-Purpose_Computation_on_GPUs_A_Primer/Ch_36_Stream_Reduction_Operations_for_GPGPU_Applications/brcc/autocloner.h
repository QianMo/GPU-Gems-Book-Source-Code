///Enhanced AutoPtr that has a proper copy constructor to allow copy constructor-style constructs to work properly with inheritance.
/** Requires class T to have a virtual Clone() function that will duplicate itself
 */
template <class T> class AutoCloner {
	///The actual pointer
	T * t;
public:
	///Default constructor sets things to NULL
	AutoCloner() {t=NULL;}
	///Pass in a T* and this sets the pointer to it.
	AutoCloner (T * t){this->t =t;}
	///The copy constructor calls the gets operator
	AutoCloner (const AutoCloner <T> &o) {*this = o;}
	///The gets operator clones T when a sole T is passed in
	AutoCloner<T> & operator = (const T &t){
		T * temp=NULL;
        this->t = t.clone(temp);
		return *this;
	}
	///When a T pointer is passed in it checks for nullity before calling Clone()
	AutoCloner<T> & operator = (const T* t){
		if (t)
			return *this = *t;
		else this->t=0;
		return *this;
	}
	///The standard gets uses one of the other gets to clone the pointer
	AutoCloner<T> & operator = (const AutoCloner <T> &o){
        return *this = o.t;
	}
	///Destructor kills this copy of T -- wow this seems like Jave
	~AutoCloner(){if (t) delete t;}
	/// Dereferences this
	const T* operator -> () const {return t;}
	/// Dereferences this	
	T* operator -> () {return t;}
	/// Checks if the pointer itself is null so that one may dereference safely
	bool IsNull() const {return t==0;}
	/// Inverse if isNull()
	bool NotNull() const {return !this->isNull();}
};

