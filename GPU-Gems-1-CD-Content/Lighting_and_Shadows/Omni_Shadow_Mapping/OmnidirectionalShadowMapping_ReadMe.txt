================================================

"Omnidirectional Shadow Mapping" Demo v1.1

================================================

CmdLine params:
---------------

-shadowmapformat : shadow map format ( f32 f16 i32pd )
-shadowmapsize   : shadow map size ( 128 256 512 1024 2048 )


Shadow Map Formats:
-------------------

f32   - D3DFMT_R32F     ( 32-bit floating-point )
f16   - D3DFMT_R16F     ( 16-bit floating-point )
i32pd - D3DFMT_A8R8G8B8 ( 32-bit integer with packed depth )


Example:
--------

OmnidirectionalShadowMapping.exe -shadowmapformat f32


Command Keys:
-------------

'Q'  - quit

'W'  - move forward
'S'  - move backward
'A'  - turn left
'D'  - turn right

UPARROW    - move forward
DOWNARROW  - move backward
LEFTARROW  - step left
RIGHTARROW - step right

'I' - invert mouse Y axis
'M' - switch mouse ON/OFF

'1' - switch wireframe mode ON/OFF 
'2' - show/hide info
'3' - toggle screen noise
'4' - switch shadows ON/OFF 
'5' - toggle default material
'7' - switch frustum culling ON/OFF
'6' - toggle PS precision 16/32 fp (only for hardware with half precision support )

'8' - switch soft shadows ON/OFF
'+' - increase shadow softness
'-' - decrease shadow softness

'9' - toggle normalization via cubemaps (only for hardware with Q8W8V8U8 cubemap support )
'0' - switch greyscale mode ON/OFF

'B' - toggle bump mapping


Contacts:
---------

philger@mail.wplus.net
