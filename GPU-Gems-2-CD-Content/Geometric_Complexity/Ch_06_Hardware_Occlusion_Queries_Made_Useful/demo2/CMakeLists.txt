PROJECT (occquery C)

ADD_EXECUTABLE(occquery DataTypes.c occquery.cpp MathStuff.c  DataTypes.h MathStuff.h HierarchyNode.h HierarchyNode.cpp Geometry.h Geometry.cpp RenderTraverser.cpp RenderTraverser.h Timers.h Timers.cpp)

IF (WIN32)
  TARGET_LINK_LIBRARIES(occquery ${CMAKE_CURRENT_SOURCE_DIR}/GL/glew32.lib)
  TARGET_LINK_LIBRARIES(occquery ${CMAKE_CURRENT_SOURCE_DIR}/GL/glut32.lib)
ELSE (WIN32)
  TARGET_LINK_LIBRARIES(occquery glew)
  TARGET_LINK_LIBRARIES(occquery glut)
ENDIF (WIN32)
