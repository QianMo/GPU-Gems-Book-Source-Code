#define STR_MINSIZE 16	//!< pString's minimum size in bytes

class pString
{
	private:
		char* buf;
		int size;
		int len;

	public:

	//! Default constructor
	pString();

	//! Construct a string from a char pointe
	pString(const char* in);

	//! Copy-constructor
	pString(const pString& in);

	//! Construct a string of length 'n' from a char pointer, beginning at position 'p'
	pString(const char* in,int p,int n);
		
	//! Default destructor
	virtual ~pString();

	//! Type cast to char pointer
	inline operator const char *() const
	{ return buf; }

	//! Indexing operator
	inline char operator[](int i) const
	{ return buf[i]; }

	//! Char pointer atribuition operator
	void operator=(const char* in);

	//! Atribuition operator
	void operator=(const pString& in);

	//! Self-concatenation operator with a char pointer
	void operator+=(const char* in);

	//! Self-concatenation operator
	void operator+=(const pString& in);

	//! Concatenation operator with a char pointer
	pString operator+(const char* in);

	//! Concatenation operator
	pString operator+(const pString& in);

	//! Copy the 'n'-sized contents of a char pointer beginning at position 'p'
	void copy(const char* in,int p,int n);

	//! Find a sub-string in the string
	int find(const char *str) const;

	//! Find the first occurrence of a character in the string
	int find(char c) const;

	//! Find the last occurrence of a character in the string
	int find_reverse(char c) const;

	//! Change the 'i'-th character of the string
	void set_char(int i,char c);

	//! Crop the first 'n' characters of the string
	void crop_begin(int n);

	//! Return a string consisting of the first 'n' characters of the original one
	pString left(int n) const;

	//! Return a string consisting of the last 'n' characters of the original one
	pString right(int n) const;

	//! Return a string consisting of the 'n' characters of the original one that follow position 'p', inclusive
	pString mid(int p,int n) const;

	//! Format the string using a format template
	void format(const char *fmt, ...);
	
	//! equal compare operator
	inline int operator==(const char *str) const 
	{ return strcmp(buf,str)==0; }

	inline int operator!=(const char *str) const 
	{ return strcmp(buf,str)!=0; }

	inline int operator>(const char *str) const 
	{ return strcmp(buf,str)>0; }

	inline int operator<(const char *str) const 
	{ return strcmp(buf,str)<0; }

	inline int operator>=(const char *str) const 
	{ return strcmp(buf,str)>=0; }

	inline int operator<=(const char *str) const 
	{ return strcmp(buf,str)<=0; }

	//! Compare with a char pointer
	inline int compare(const char *str) const
	{ return strcmp(buf,str); }

	//! Compare the first 'n' characters of the string with a char pointer
	inline int compare(const char *str,int n) const
	{ return strncmp(buf,str,n); }

	//! Compare with a char pointer, case-insensitive flavour
	inline int compare_nocase(const char *str) const
	{ return stricmp(buf,str); }

	//! Change all characters to lower-case
	inline void lower()
	{ strlwr(buf); }

	//! Change all characters to upper-case
	inline void upper()
	{ strupr(buf); }

	//! Return the length of the string in bytes
	inline int length() const
	{ return len; }

	//! allocs a fixed length string of s bytes
	void reserve(int s);
	
	// writes the string bytes to a file
	int write(FILE *fp) const;
	
	// reads the string bytes from a file
	int read(FILE *fp);
};

typedef pArray<pString> pStringArray;
typedef pArray<pStringArray> pStringArray2;
