# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cap/code/lidar_descriptor/LiDAR-Iris-test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cap/code/lidar_descriptor/LiDAR-Iris-test/build

# Include any dependencies generated for this target.
include CMakeFiles/demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo.dir/flags.make

CMakeFiles/demo.dir/demo.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/demo.cpp.o: ../demo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cap/code/lidar_descriptor/LiDAR-Iris-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo.dir/demo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/demo.cpp.o -c /home/cap/code/lidar_descriptor/LiDAR-Iris-test/demo.cpp

CMakeFiles/demo.dir/demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/demo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cap/code/lidar_descriptor/LiDAR-Iris-test/demo.cpp > CMakeFiles/demo.dir/demo.cpp.i

CMakeFiles/demo.dir/demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/demo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cap/code/lidar_descriptor/LiDAR-Iris-test/demo.cpp -o CMakeFiles/demo.dir/demo.cpp.s

CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.o: ../LidarIris/LidarIris.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cap/code/lidar_descriptor/LiDAR-Iris-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.o -c /home/cap/code/lidar_descriptor/LiDAR-Iris-test/LidarIris/LidarIris.cpp

CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cap/code/lidar_descriptor/LiDAR-Iris-test/LidarIris/LidarIris.cpp > CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.i

CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cap/code/lidar_descriptor/LiDAR-Iris-test/LidarIris/LidarIris.cpp -o CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.s

CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.o: ../LidarIris/fftm/fftm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cap/code/lidar_descriptor/LiDAR-Iris-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.o -c /home/cap/code/lidar_descriptor/LiDAR-Iris-test/LidarIris/fftm/fftm.cpp

CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cap/code/lidar_descriptor/LiDAR-Iris-test/LidarIris/fftm/fftm.cpp > CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.i

CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cap/code/lidar_descriptor/LiDAR-Iris-test/LidarIris/fftm/fftm.cpp -o CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.s

# Object files for target demo
demo_OBJECTS = \
"CMakeFiles/demo.dir/demo.cpp.o" \
"CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.o" \
"CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.o"

# External object files for target demo
demo_EXTERNAL_OBJECTS =

demo: CMakeFiles/demo.dir/demo.cpp.o
demo: CMakeFiles/demo.dir/LidarIris/LidarIris.cpp.o
demo: CMakeFiles/demo.dir/LidarIris/fftm/fftm.cpp.o
demo: CMakeFiles/demo.dir/build.make
demo: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_people.so
demo: /usr/lib/x86_64-linux-gnu/libboost_system.so
demo: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
demo: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
demo: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
demo: /usr/lib/x86_64-linux-gnu/libboost_regex.so
demo: /usr/lib/x86_64-linux-gnu/libqhull.so
demo: /usr/lib/libOpenNI.so
demo: /usr/lib/libOpenNI2.so
demo: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libfreetype.so
demo: /usr/lib/x86_64-linux-gnu/libz.so
demo: /usr/lib/x86_64-linux-gnu/libjpeg.so
demo: /usr/lib/x86_64-linux-gnu/libpng.so
demo: /usr/lib/x86_64-linux-gnu/libtiff.so
demo: /usr/lib/x86_64-linux-gnu/libexpat.so
demo: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
demo: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_features.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_search.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_io.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
demo: /usr/lib/x86_64-linux-gnu/libpcl_common.so
demo: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libz.so
demo: /usr/lib/x86_64-linux-gnu/libGLEW.so
demo: /usr/lib/x86_64-linux-gnu/libSM.so
demo: /usr/lib/x86_64-linux-gnu/libICE.so
demo: /usr/lib/x86_64-linux-gnu/libX11.so
demo: /usr/lib/x86_64-linux-gnu/libXext.so
demo: /usr/lib/x86_64-linux-gnu/libXt.so
demo: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
demo: /usr/lib/x86_64-linux-gnu/libfreetype.so
demo: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
demo: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
demo: CMakeFiles/demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cap/code/lidar_descriptor/LiDAR-Iris-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo.dir/build: demo

.PHONY : CMakeFiles/demo.dir/build

CMakeFiles/demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo.dir/clean

CMakeFiles/demo.dir/depend:
	cd /home/cap/code/lidar_descriptor/LiDAR-Iris-test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cap/code/lidar_descriptor/LiDAR-Iris-test /home/cap/code/lidar_descriptor/LiDAR-Iris-test /home/cap/code/lidar_descriptor/LiDAR-Iris-test/build /home/cap/code/lidar_descriptor/LiDAR-Iris-test/build /home/cap/code/lidar_descriptor/LiDAR-Iris-test/build/CMakeFiles/demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo.dir/depend

