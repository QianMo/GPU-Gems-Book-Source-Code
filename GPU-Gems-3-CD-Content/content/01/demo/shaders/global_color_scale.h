// if your rendertargets are rgb10a2
// then you might want to multiply all colors by 0.5 before writing them
// this gives us an effective range of 0..2 [giving some HDR-ness]
// then in composite_f we scale back down to 0..1 range.

#define GLOBAL_COLOR_SCALE   ( 1 )   // should be < 1.  Ex: 0.5
