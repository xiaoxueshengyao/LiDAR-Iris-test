cmake_minimum_required (VERSION 3.8)

SET(CMAKE_BUILD_TYPE Release)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu/cmake")
set(OpenCV_DIR "/usr/local/share/OpenCV")

project (LidarIris)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package ( OpenMP )
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories("./LidarIris/")
include_directories("./Scancontext/")
include_directories(/opt/ros/noetic/include)

add_executable(demo "demo.cpp" "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
target_link_libraries(demo ${PCL_LIBRARIES} ${OpenCV_LIBS})


# add_executable(front "front_view.cpp")
# target_link_libraries(front ${PCL_LIBRARIES} ${OpenCV_LIBS})


# add_executable(test_icp "test_icp.cpp" "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
# target_link_libraries(test_icp ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(demo2 demo2.cpp  Scancontext/Scancontext.cpp Scancontext/nanoflann.hpp "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
target_link_libraries(demo2 ${PCL_LIBRARIES} ${OpenCV_LIBS})


add_executable(demo3 demo3.cpp  Scancontext/Scancontext.cpp Scancontext/nanoflann.hpp)
target_link_libraries(demo3 PUBLIC OpenMP::OpenMP_CXX ${PCL_LIBRARIES} ${OpenCV_LIBS})