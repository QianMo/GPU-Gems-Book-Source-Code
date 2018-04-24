#ifndef _SLIDERS_H_
#define _SLIDERS_H_ 1

  cbuffer SliderCB {
    // note: if you add/change things here,
    //       change them in models\main.nma as well.
    float bump_scale;   
    float bump_freq;   
    float fog_density;  
    float diffuse_light;
    float spec_light;
    float light_wrap;
    float ambient_occlusion;
    float color_saturation;
    //float my_new_value_1;
    //float my_new_value_2;
    //float my_new_value_3;
  };

#endif //_SLIDERS_H_