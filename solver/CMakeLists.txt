cmake_minimum_required(VERSION 3.0)
project(APASBackend)

FIND_PACKAGE(Boost 1.58 COMPONENTS program_options regex graph REQUIRED)



set(EXT_DIR "${CMAKE_CURRENT_BINARY_DIR}/third-party/ExternalProject")
message(STATUS "EXT_DIR: ${EXT_DIR}")

if(MADPPATH)
	set(MADP_INCLUDE_DIRECTORIES ${MADPPATH}/include)
	set(MADP_LIBRARY_PATH ${MADPPATH}/lib)
else()
	include(ExternalProject)
	ExternalProject_Add(project_madp
		GIT_REPOSITORY	   https://github.com/laurimi/MADP
		GIT_TAG			   75b4cec91ac03e5bb85d4b5328601178daee1ba7
	    PREFIX            ${EXT_DIR}/madp
	    BUILD_IN_SOURCE	  1
	    CONFIGURE_COMMAND sh <SOURCE_DIR>/autogen.sh && <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
	    BUILD_COMMAND make
	    INSTALL_COMMAND make install
	)

	ExternalProject_Get_Property(project_madp INSTALL_DIR BINARY_DIR SOURCE_DIR)
    set(MADP_INCLUDE_DIRECTORIES ${INSTALL_DIR}/include ${SOURCE_DIR}/src/include)
	set(MADP_LIBRARY_PATH ${INSTALL_DIR}/lib)

	message(STATUS "MADP_INCLUDE_DIRECTORIES: ${MADP_INCLUDE_DIRECTORIES}")
	message(STATUS "MADP_LIBRARY_PATH: ${MADP_LIBRARY_PATH}")
	add_library(MADP STATIC IMPORTED)
	set_property(TARGET MADP PROPERTY IMPORTED_LOCATION ${MADP_LIBRARY_PATH}/libMADP-0.4.1.so)

	add_dependencies(MADP project_madp)
	
endif()


include_directories(include
					SYSTEM ${MADP_INCLUDE_DIRECTORIES}
					)
link_directories(${MADP_LIBRARY_PATH})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

#find_package(PkgConfig)
#pkg_search_module(Eigen3 REQUIRED eigen3)

# Ref: https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_definitions(
    -Wfatal-errors
    )

# Add warnings and errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

add_library(MADPWrapper src/madp_wrapper/MADPWrapper.cpp src/madp_wrapper/MADPWrapperUtilities.cpp)
target_link_libraries(MADPWrapper MADP)


add_executable(value_fsc_policy
				src/utils/value_fsc_policy.cpp
				)
target_link_libraries(value_fsc_policy ${Boost_LIBRARIES} MADPWrapper Eigen3::Eigen)

add_executable(fsc_policy_endstates
				src/utils/fsc_policy_endstates.cpp
				)
target_link_libraries(fsc_policy_endstates ${Boost_LIBRARIES} MADPWrapper Eigen3::Eigen)

add_executable(improve_fsc
				src/utils/improve_fsc.cpp
				)
target_link_libraries(improve_fsc ${Boost_LIBRARIES} MADPWrapper Eigen3::Eigen)

add_executable(improve_fsc_preds
				src/utils/improve_fsc_preds.cpp
				)
target_link_libraries(improve_fsc_preds ${Boost_LIBRARIES} MADPWrapper Eigen3::Eigen)
