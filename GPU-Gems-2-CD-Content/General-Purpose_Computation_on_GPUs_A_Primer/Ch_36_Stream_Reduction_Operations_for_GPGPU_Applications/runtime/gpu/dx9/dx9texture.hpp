// dx9texture.hpp
#pragma once

#include "dx9base.hpp"

namespace brook
{

    class DX9Texture
    {
    public:
        enum ComponentType
        {
          kComponentType_Float = 0,
          kComponentType_Fixed = 1,
          kComponentType_Half = 2
        };

        static DX9Texture* create(
          GPUContextDX9* inContext,
          int inWidth, int inHeight,
          int inComponents,
          ComponentType inComponentType = kComponentType_Float );
        ~DX9Texture();

        int getWidth() { return width; }
        int getHeight() { return height; }

        void setData( const float* inData, unsigned int inStride, unsigned int inCount,
          unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
          const unsigned int* inExtents, bool inUsesAddressTranslation );
        void getData( float* outData, unsigned int inStride, unsigned int inCount,
          unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
          const unsigned int* inExtents, bool inUsesAddressTranslation );

        void markCachedDataChanged();
        void markShadowDataChanged();
        void validateCachedData();
        void validateShadowData();

        void getPixelAt( int x, int y, float4& outResult );

        LPDIRECT3DTEXTURE9 getTextureHandle() {
        return textureHandle;
        }

        LPDIRECT3DSURFACE9 getSurfaceHandle() {
        return surfaceHandle;
        }

    private:
      DX9Texture( GPUContextDX9* inContext, int inWidth, int inHeight, int inComponents,
        ComponentType inComponentType = kComponentType_Float  );
        bool initialize();

        void flushCachedToShadow();
        void flushShadowToCached();
        void getShadowData( void* outData, unsigned int inStride, unsigned int inCount,
          unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
          const unsigned int* inExtents, bool inUsesAddressTranslation );
        void setShadowData( const void* inData, unsigned int inStride, unsigned int inCount,
          unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
          const unsigned int* inExtents, bool inUsesAddressTranslation );

        void findRectForCopy( unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
          const unsigned int* inExtents, bool inUsesAddressTranslation, RECT& outRect, bool& outFullBuffer );
        void copyData( void* toBuffer, size_t toRowStride, size_t toElementStride,
                       const void* fromBuffer, size_t fromRowStride, size_t fromElementStride,
                       size_t columnCount, size_t rowCount, size_t elementCount, size_t elementSize );
        void findRectForCopyAT( unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
          const unsigned int* inExtents, bool inUsesAddressTranslation, RECT& outRect, bool& outFullBuffer,
          size_t inWidth, size_t inHeight, size_t& outBaseX, size_t& outBaseY );
        void copyAllDataAT( void* toBuffer, size_t toRowStride, size_t toElementStride,
                            const void* fromBuffer, size_t fromRowStride,size_t fromElementStride,
                            size_t columnCount, size_t rowCount, size_t elementCount, size_t elementSize, size_t inRank, const unsigned int* inExtents );

        void getDataAT( void* streamData, unsigned int streamRank,
          const unsigned int* streamDomainMin, const unsigned int* streamDomainMax,
          const unsigned int* streamExtents, size_t streamElementStride,
          const void* textureData, size_t textureLineStride, size_t textureWidth, size_t textureHeight,
          size_t textureElementStride, size_t textureBaseX, size_t textureBaseY );
        void setDataAT( const void* streamData, unsigned int streamRank,
          const unsigned int* streamDomainMin, const unsigned int* streamDomainMax,
          const unsigned int* streamExtents, size_t streamElementStride,
          void* textureData, size_t textureLineStride, size_t textureWidth, size_t textureHeight,
          size_t textureElementStride, size_t textureBaseX, size_t textureBaseY );

        GPUContextDX9* _context;
        LPDIRECT3DDEVICE9 device;

        int width;
        int height;
        int components;
        int internalComponents;
        ComponentType componentType;
        int componentSize;
        LPDIRECT3DTEXTURE9 textureHandle;
        LPDIRECT3DSURFACE9 surfaceHandle;

        LPDIRECT3DSURFACE9 shadowSurface;

        enum DirtyFlag {
            kShadowDataDirty = 0x01,
            kCachedDataDirty = 0x02
        };
        int dirtyFlags;
    };
}
