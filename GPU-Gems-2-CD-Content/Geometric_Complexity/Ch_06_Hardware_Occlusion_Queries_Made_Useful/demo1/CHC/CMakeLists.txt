PROJECT (CHC C)

SET(COMMON ${CMAKE_CURRENT_SOURCE_DIR}/../!common)
SET(MATH ${COMMON}/Mathematic)
SET(ENGINE ${COMMON}/Engine)

INCLUDE_DIRECTORIES(${COMMON})

ADD_EXECUTABLE(CHC DataTypes.cpp LispSM.cpp Main.cpp MathStuff.cpp FrameState.h SceneData.cpp DataTypes.h Main.h MathStuff.h SceneData.h Geometry.h Geometry.cpp GeometryObjects.h Occlusion.h Occlusion.cpp)

ADD_LIBRARY(math STATIC 
  ${MATH}/Mathematic.cpp ${MATH}/MathTools.cpp ${MATH}/Matrix4.cpp
  ${MATH}/Perlin.cpp ${MATH}/Vector3.cpp ${MATH}/Vector4.cpp
)

ADD_LIBRARY(engine STATIC 
  ${ENGINE}/GLInterface/PBuffer.cpp
  ${ENGINE}/Renderer/State.cpp
  ${ENGINE}/WindowManager/KeyState.cpp
)


TARGET_LINK_LIBRARIES(CHC math)
TARGET_LINK_LIBRARIES(CHC engine)

IF (WIN32)
  TARGET_LINK_LIBRARIES(CHC ${COMMON}/GL/glew32.lib)
  TARGET_LINK_LIBRARIES(CHC ${COMMON}/GL/glut32.lib)
ELSE (WIN32)
  TARGET_LINK_LIBRARIES(CHC glew)
  TARGET_LINK_LIBRARIES(CHC glut)
ENDIF (WIN32)


