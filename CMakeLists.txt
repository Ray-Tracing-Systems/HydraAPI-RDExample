cmake_minimum_required(VERSION 3.7)
project(hydra_api_tests)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
        Camera.h
        input.h input.cpp
        Timer.h Timer.cpp
        mesh_utils.h mesh_utils.cpp
        main.cpp
        window_main_free_look_gl.cpp)

set(SOURCE_FILES_GL
        RenderDrivers.cpp
        RenderDriverOpenGL1.h
        RenderDriverOpenGL1.cpp
        )

#set (CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}")

set(HYDRA_API_ROOT_DIR ${CMAKE_SOURCE_DIR}/../HydraAPI CACHE PATH "Path to HydraAPI") #
set(HYDRA_API_SRC_DIR ${HYDRA_API_ROOT_DIR}/hydra_api)
set(HYDRA_API_LIB_DIR ${HYDRA_API_ROOT_DIR}/bin)
set(HYDRA_API_UTILS_SRC_DIR ${HYDRA_API_ROOT_DIR}/utils)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
else()
  find_package(glfw3 REQUIRED)
  include_directories(${GLFW_INCLUDE_DIRS})
  set(LIBS ${LIBS} ${GLFW_LIBRARIES})

  find_package (Threads REQUIRED)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()


find_package(OpenGL REQUIRED)
add_executable(hydra_api_tests ${SOURCE_FILES} ${SOURCE_FILES_GL})
include_directories(${ADDITIONAL_INCLUDE_DIRS} ${HYDRA_API_SRC_DIR} ${HYDRA_API_UTILS_SRC_DIR})
find_library(HYDRA_API_LIB hydra_api HINTS ${HYDRA_API_LIB_DIR} NO_DEFAULT_PATH)
find_library(HYDRA_API_IES_PARSER_LIB ies_parser HINTS ${HYDRA_API_LIB_DIR} NO_DEFAULT_PATH)
find_library(HYDRA_API_MIKKTSPACE mikktspace HINTS ${HYDRA_API_LIB_DIR} NO_DEFAULT_PATH)
find_library(HYDRA_API_CORTO corto HINTS ${HYDRA_API_LIB_DIR} NO_DEFAULT_PATH)

find_package(FreeImage REQUIRED)
include_directories(${FREEIMAGE_INCLUDE_DIRS})
set(LIBS ${LIBS} ${FREEIMAGE_LIBRARIES})


target_include_directories(hydra_api_tests PUBLIC ${OPENGL_INCLUDE_DIR})

if(WIN32)
  add_definitions(-DUNICODE -D_UNICODE)
  add_definitions(-DNEED_DIR_CHANGE)
  target_link_libraries(hydra_api_tests LINK_PUBLIC ${OPENGL_gl_LIBRARY} hydra_api glfw3dll )
else()
  target_compile_options(hydra_api_tests PRIVATE -fpermissive -Wnarrowing)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fno-omit-frame-pointer -fsanitize=leak)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined")
    MESSAGE( STATUS "WE USE CLANG(main);")
  else()
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")
    MESSAGE( STATUS "WE USE GCC  (main);")
  endif()

  MESSAGE( STATUS "CMAKE_CXX_FLAGS(main): " ${CMAKE_CXX_FLAGS} )
  target_link_libraries(hydra_api_tests LINK_PUBLIC
          ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY}
          ${HYDRA_API_LIB}
          ${LIBS}
          ${HYDRA_API_IES_PARSER_LIB} ${HYDRA_API_MIKKTSPACE} ${HYDRA_API_CORTO}
          Threads::Threads stdc++fs rt dl OpenCL glfw)

endif()


