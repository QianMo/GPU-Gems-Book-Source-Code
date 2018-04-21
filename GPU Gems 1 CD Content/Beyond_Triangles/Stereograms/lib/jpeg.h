typedef struct 
{
   unsigned char *ptr;
   int    width;
   int    height;
   FILE  *output_file;
   FILE  *input_file;
   int    aritcoding;
   int    CCIR601sampling;
   int    smoothingfactor;
   int    quality;
   int    status;
   int    components;
   int    numscan;
} JPEGDATA;

void jpeg_info(JPEGDATA *data);
void jpeg_read(JPEGDATA *data);
void jpeg_write(JPEGDATA *data);
