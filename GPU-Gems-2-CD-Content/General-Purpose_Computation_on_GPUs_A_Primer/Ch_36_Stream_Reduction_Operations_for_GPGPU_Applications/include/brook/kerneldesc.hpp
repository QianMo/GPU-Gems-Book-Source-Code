// kerneldesc.hpp
#ifndef KERNELDESC_HPP
#define KERNELDESC_HPP

#include <vector>

namespace brook
{
  namespace desc
  {
    // Globals
    enum
    {
      kGlobalConstant_Workspace = 0
    };
    enum
    {
      kGlobalConstant_ATOutputLinearize = 0,
      kGlobalConstant_ATOutputStride,
      kGlobalConstant_ATOutputInvStride,
      kGlobalConstant_ATOutputInvExtent,
      kGlobalConstant_ATOutputDomainMin,
      kGlobalConstant_ATOutputDomainSize,
      kGlobalConstant_ATOutputInvShape,
      kGlobalConstant_ATHackConstant
    };
    enum
    {
      kGlobalInterpolant_ATOutputTex = 0,
      kGlobalInterpolatn_ATOutputAddress
    };

    // Stream Arguments
    enum
    {
      kStreamConstant_Indexof = 0,
      kStreamConstant_ATIndexofNumer,
      kStreamConstant_ATIndexofDenom,
      kStreamConstant_ATLinearize,
      kStreamConstant_ATTextureShape,
      kStreamConstant_ATDomainMin
    };
    enum
    {
      kStreamInterpolant_Position = 0
    };

    // Gather Arguments
    enum
    {
      kGatherConstant_Shape = 0,
      kGatherConstant_ATLinearize,
      kGatherConstant_ATTextureShape,
      kGatherConstant_ATDomainMin
    };

    // Iterator Arguments
    enum
    {
      kIteratorInterpolant_Value = 0
    };
    enum
    {
      kIteratorConstant_ATIndexofNumer = 0,
      kIteratorConstant_ATIndexofDenom,
      kIteratorConstant_ATValueBase,
      kIteratorConstant_ATValueOffset1,
      kIteratorConstant_ATValueOffset4
    };

    // Output Arguments
    enum
    {
      kOutputConstant_Indexof = 0
    };

    enum
    {
      kOutputInterpolant_Position = 0
    };

    class gpu_input_desc
    {
    public:
      gpu_input_desc( int argumentIndex, int componentIndex )
        : _argumentIndex(argumentIndex), _componentIndex(componentIndex)
      {}

      int _argumentIndex;
      int _componentIndex;
    };

    class gpu_pass_desc
    {
    public:
      gpu_pass_desc( const char* shaderString )
        : _shaderString(shaderString)
      {}

      gpu_pass_desc& constant( int argument, int component ) {
        constants.push_back( gpu_input_desc(argument, component) );
        return *this;
      }

      gpu_pass_desc& sampler( int argument, int component ) {
        samplers.push_back( gpu_input_desc(argument, component) );
        return *this;
      }

      gpu_pass_desc& interpolant( int argument, int component ) {
        interpolants.push_back( gpu_input_desc(argument, component) );
        return *this;
      }

      gpu_pass_desc& output( int argument, int component ) {
        outputs.push_back( gpu_input_desc(argument, component) );
        return *this;
     }

      const char* _shaderString;
      std::vector<gpu_input_desc> constants;
      std::vector<gpu_input_desc> samplers;
      std::vector<gpu_input_desc> interpolants;
      std::vector<gpu_input_desc> outputs;
    };

    class gpu_technique_desc
    {
    public:
      gpu_technique_desc()
        : _reductionFactor(-1), _outputAddressTranslation(false), _inputAddressTranslation(false), _temporaryCount(0)
      {}

      gpu_technique_desc& pass( const gpu_pass_desc& p ) {
        _passes.push_back( p );
        return *this;
      }

      gpu_technique_desc& reduction_factor( int factor ) {
        _reductionFactor = factor;
        return *this;
      }

      gpu_technique_desc& output_address_translation() {
        _outputAddressTranslation = true;
        return *this;
      }

      gpu_technique_desc& input_address_translation() {
        _inputAddressTranslation = true;
        return *this;
      }

      gpu_technique_desc& temporaries( int temporaryCount ) {
        _temporaryCount = temporaryCount;
        return *this;
      }

      int _reductionFactor;
      bool _outputAddressTranslation;
      bool _inputAddressTranslation;
      int _temporaryCount;
      std::vector< gpu_pass_desc > _passes;
    };

    class gpu_kernel_desc
    {
    public:
      gpu_kernel_desc& technique( const gpu_technique_desc& t ) {
        _techniques.push_back( t );
        return *this;
      }

      std::vector< gpu_technique_desc > _techniques;
    };
  };
};

#endif
