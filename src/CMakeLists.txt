function(add_example basename)
  cmake_minimum_required(VERSION 3.8s)
  # set C++ standard to C++17
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_BUILD_TYPE "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


  # add WITH_EIGEN, if specified
  if(WITH_EIGEN)
    add_definitions("-DWITH_EIGEN")
  endif()

  # add the exectuable and link it to the Python libs
  add_executable(${basename} ${basename}.cpp ../matplotlibcpp.h)
  target_link_libraries(${basename} ${Python3_LIBRARIES})
endfunction(add_example)

# add the executables
add_example(HW1)

