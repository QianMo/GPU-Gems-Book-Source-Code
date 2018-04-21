class pPicture
{
	public:
		char name[256];		//!< picture filename
		int sx,				//!< x size in pixels 
			sy,				//!< y size in pixels 
			sz,				//!< z size in pixels 
			bytespixel,		//!< number of bytes per pixel (24 or 32)
			size;			//!< size in bytes for the image (sx*sy*bytespixel)
		unsigned char *buf;	//!< image pixels 
		int normalmapflag;	//!< is image a normal map

	//! Default constructor
	pPicture() :
		buf(0),sx(0),sy(0),sz(0),
		bytespixel(0),size(0),
		normalmapflag(0)
	{ 	
		name[0]=0; 
	}

	//! Copy-constructor
	pPicture(pPicture& in);

	//! atribuition operator
	void operator=(const pPicture& in);

	//! Default Destructor
	virtual ~pPicture()
	{ FreePicture(); }

	// Loads a .tga or .jpg picture from a file
	int LoadPIC(const char *file,int flipy=0);

	// Loads a .tga picture from a file
	int LoadTGA(const char *file,int flipy=0);
	// Loads a .jpg picture from a file
	int LoadJPG(const char *file,int flipy=0);
	
	// Loads a .t3d picture from a file
	int LoadT3D(const char *file);
	//! Load a .t3d image from memory buffer 
	int LoadT3D(const unsigned char *data,int len);

	//! Load a .tga image. Only 24 or 32 bits/pixel images are supported, uncompressed or RLE compressed
	int LoadTGA(const unsigned char *data,int len,int flipy=0);
	//! Save a .tga image
	int SaveTGA(const char *file);

	//! Save a .jpg image at specified quality factor (0-100)
	int SaveJPG(const char *file,int quality=85,int progressive=0);
	//! Save a .jpg image at specified quality factor (0-100)
	int SaveJPG(FILE *fp,int quality=85,int progressive=0);

	//! Allocate the memory required for a picture with the specified dimensions and color depth. The buffer allocated is xc*yd*bp bytes
	void CreatePicture(int bp,int xd,int yd,int zd=1);
	//! Allocate the memory required for the 32-bit picture with the specified dimensions. The buffer allocated is xc*yd*4 bytes
	void CreatePicture32(int xd,int yd);
	//! Allocate the memory required for the 24-bit picture with the specified dimensions. The buffer allocated is xc*yd*3 bytes
	void CreatePicture24(int xd,int yd);

	//! Free all data allocated by the picture
	void FreePicture(void);

	//! Resamples the image to the closest power of 2 dimensions
	void CheckSize(int droplevel);
	//! Flips picture in Y axis (first line turns to last line)
	void FlipY();

	void GetPixel(float x,float y,pVector& rgb,int texfilter) const;
	void GetPixel(int x,int y,pVector& rgb) const;

	inline void pPicture::GetPixelUV(float x,float y,pVector& rgb,int texfilter) const
	{ GetPixel(x*sy,y*sy,rgb,texfilter); }
	void GetPixelDxDy(float x,float y,float& dx,float& dy,int texfilter) const;

	void ToNormalMap(pPicture& p,float factor=1.0f);
	int GetGrayPixel(int x,int y) const
	{
		unsigned char *uc=&buf[(((x+sx)%sx)+((y+sy)%sy)*sx)*bytespixel];
		return ((int)uc[0]+(int)uc[1]+(int)uc[2])/3;
	}

	int PixelDifference(int x1,int y1,int x2,int y2);
	int ShouldAntiAlias(int x,int y,int factor);
};

