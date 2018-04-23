// gpustream.hpp
#ifndef GPU_STREAM_HPP
#define GPU_STREAM_HPP

#include "gpucontext.hpp"

namespace brook {

    class GPUStreamData
    {
    public:
        typedef GPUContext::TextureFormat TextureFormat;
        typedef GPUContext::TextureHandle TextureHandle;

        GPUStreamData( GPURuntime* inRuntime );

        bool initialize(
            unsigned int inFieldCount, 
            const StreamType* inFieldTypes,
            unsigned int inDimensionCount, 
            const unsigned int* inExtents );

        void acquireReference();
        void releaseReference();

        void setData( const void* inData );
        void getData( void* outData );

        void setDomainData( const void* inData, const unsigned int* inDomainMin, const unsigned int* inDomainMax );
        void getDomainData( void* outData, const unsigned int* inDomainMin, const unsigned int* inDomainMax );
        void* map(unsigned int flags);
        void unmap(unsigned int flags);

        const unsigned int* getExtents() const { return &_extents[0]; }
        unsigned int getRank() const { return _rank; }
        unsigned int getTotalSize() const { return _totalSize; }
        unsigned int getFieldCount() const { return _fields.size(); }
        StreamType getIndexedFieldType(unsigned int i) const {
            return _fields[i].elementType;
        }
        unsigned int getElementSize() const {
            return _elementSize;
        }

        TextureHandle getIndexedFieldTexture( size_t inIndex ) {
            return _fields[inIndex].texture;
        }

        float4 getIndexofConstant() const { return _indexofConstant; }

        unsigned int getTextureWidth() const {return _textureWidth; }
        unsigned int getTextureHeight() const {return _textureHeight; }
        const unsigned int* getReversedExtents() const { return &_reversedExtents[0]; }
        bool requiresAddressTranslation() const { return _requiresAddressTranslation; }

        void getOutputRegion(
            const unsigned int* inDomainMin,
            const unsigned int* inDomainMax,
            GPURegion &outRegion );


        void getStreamInterpolant(
            const unsigned int* inDomainMin,
            const unsigned int* inDomainMax,
            unsigned int inOutputWidth,
            unsigned int inOutputHeight,
            GPUInterpolant &outInterpolant );

        ~GPUStreamData();

    private:
      friend class GPUStream;

        class Field
        {
        public:
            Field( StreamType inElementType, 
                   unsigned int inComponentCount, 
                   TextureHandle inTexture )
                : elementType(inElementType), 
                  componentCount(inComponentCount), 
                  texture(inTexture)
            {}

            StreamType elementType;
            unsigned int componentCount;
            TextureHandle texture;
        };

        GPURuntime* _runtime;
        GPUContext* _context;

        unsigned int _rank;
        unsigned int _totalSize;
        std::vector<unsigned int> _extents;
        std::vector<unsigned int> _reversedExtents;
        std::vector<Field> _fields;
        unsigned int _elementSize;

        float4 _indexofConstant;
        unsigned int _textureWidth, _textureHeight;
        GPUInterpolant _defaultInterpolant;
        GPURegion _outputRegion;

        unsigned int _referenceCount;
        void* _cpuData;
        size_t _cpuDataSize;

        bool _requiresAddressTranslation;

    };

    class GPUStream :
        public Stream
    {
    public:
        typedef GPUContext::TextureFormat TextureFormat;
        typedef GPUContext::TextureHandle TextureHandle;

        static GPUStream* GPUStream::create( GPURuntime* inRuntime,
                        unsigned int inFieldCount, 
                        const StreamType* inFieldTypes,
                        unsigned int inDimensionCount, 
                        const unsigned int* inExtents );

        GPUStream( GPUStreamData* inData );
        GPUStream( GPUStreamData* inData,
            const unsigned int* inDomainMin,
            const unsigned int* inDomainMax );

        virtual void Read( const void* inData ) {
            _data->setDomainData( inData, _domainMin, _domainMax );
        }

        virtual void Write( void* outData ) {
            _data->getDomainData( outData, _domainMin, _domainMax );
        }

        virtual Stream* Domain(int min, int max);
        virtual Stream* Domain(const int2& min, const int2& max);
        virtual Stream* Domain(const int3& min, const int3& max);
        virtual Stream* Domain(const int4& min, const int4& max);

      virtual const unsigned int * getDomainMin() const {
        return _domainMin;
      }

      virtual const unsigned int * getDomainMax() const {
        return _domainMax;
      }

        virtual void* getData (unsigned int flags);
        virtual void releaseData(unsigned int flags);

        virtual const unsigned int* getExtents() const {
            return _data->getExtents();
        }
        
        virtual unsigned int getDimension() const {
            return _data->getRank();
        }
        
        virtual unsigned int getTotalSize() const {
            return _data->getTotalSize();
        }
        
        virtual unsigned int getFieldCount() const {
            return _data->getFieldCount();
        }

        virtual StreamType getIndexedFieldType(unsigned int i) const {
            return _data->getIndexedFieldType(i);
        }

        TextureHandle getIndexedFieldTexture( size_t inIndex ) {
            return _data->getIndexedFieldTexture(inIndex);
        }

        float4 getIndexofConstant() const {
            return _data->getIndexofConstant();
        }

        float4 getGatherConstant() const;

        unsigned int getTextureWidth() const {
            return _data->getTextureWidth();
        }

        unsigned int getTextureHeight() const {
            return _data->getTextureHeight();
        }

        unsigned int getRank() const {
            return _data->getRank();
        }

        const unsigned int* getReversedExtents() const {
            return _data->getReversedExtents();
        }

        bool requiresAddressTranslation() const {
            return _data->requiresAddressTranslation();
        }

        void getOutputRegion( GPURegion& outRegion );

        void getStreamInterpolant( unsigned int _textureWidth,
                                   unsigned int _textureHeight,
                                   GPUInterpolant &_interpolant );

        float4 getATLinearizeConstant();
        float4 getATTextureShapeConstant();
        float4 getATDomainMinConstant();

        // TIM: hacky magic stuff for rendering
        void* getIndexedFieldRenderData(unsigned int i);
        void   synchronizeRenderData();


  private:
        virtual ~GPUStream ();

        enum {
            kMaximumRank = 4
        };

        GPUStreamData* _data;
        unsigned int _domainMin[kMaximumRank];
        unsigned int _domainMax[kMaximumRank];
    };

}

#endif
