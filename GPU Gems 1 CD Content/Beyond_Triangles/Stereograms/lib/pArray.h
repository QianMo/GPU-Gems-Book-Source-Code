template <class T> 
class pArray
{
public:
	T *buf;		//!< generic buffer
	int num;	//!< number of entries
	
	int arraysize;	//!< pArray size in bytes

	//! Default constructor
	pArray() : 
		arraysize(0), buf(0), num(0)
	{ }

	//! Copy-constructor
	pArray(const pArray<T>& in) :
		arraysize(0), buf(0), num(0)
	{ 
		reserve(in.num);
		for( int i=0;i<in.num;i++ )
			add(in[i]);
	}

	//! Default destructor
	virtual ~pArray()
	{ delete[] buf; }

	//! Copy-constructor
	void operator=(const pArray<T>& in)
	{ 
		clear();
		reserve(in.num);
		for( int i=0;i<in.num;i++ )
			add(in[i]);
	}

	//! Adds another pArray to the end of current pArray
	virtual void operator+=(pArray<T>& in)
	{
		if (num+in.num>arraysize)
		{
			arraysize=num+in.num;
			T *tmp=new T[arraysize];
			for( int i=0;i<num;i++ )
				tmp[i]=buf[i];
			delete[] buf;
			buf=tmp;
		}
		for( int i=0;i<in.num;i++ )
			buf[i+num]=in[i];
		num+=in.num;
	}

	//! Reserve the required amount of space for the pArray
	virtual void reserve(int n)
	{
		if (arraysize<n)
		{
			arraysize=n;
			T *tmp=new T[arraysize];
			for( int i=0;i<num;i++ )
				tmp[i]=buf[i];
			delete[] buf;
			buf=tmp;
		}
	}

	//! Free all the space
	virtual void free()
	{
		delete[] buf;
		buf=0;
		num=0;
		arraysize=0;
	}

	//! Clear the whole pArray, but keep the memory space
	inline void clear()
	{
		num=0;
	}

	//! Add a new element to the end of the pArray, automatically allocating more space, if needed
	virtual void add(T elem)
	{
		if (num==arraysize)
		{
			if (arraysize==0)
				arraysize=4;
			else
				arraysize+=arraysize>>1;
			T *tmp=new T[arraysize];
			for( int i=0;i<num;i++ )
				tmp[i]=buf[i];
			delete[] buf;
			buf=tmp;
		}
		buf[num++]=elem;
	}

	//! Remove the element in the given position
	inline void remove(int i)
	{
		if (i<num)
		{
			int j;
			for (j = i; j < num-1; j++) 
				buf[j] = buf[j+1];
			num--;
		}
	}

	//! Remove 'n' elements in the given position
	inline void remove(int i,int n)
	{
		if (i+n<=num)
		{
			int j;
			for (j = i; j < num-n; j++) 
				buf[j] = buf[j+n];
			num-=n;
		}
	}

	//! Indexing operator returing const
	inline const T& operator[](int i) const 
	{ return(buf[i]); }

	//! Indexing operator
	inline T& operator[](int i) 
	{ return buf[i]; }
};
