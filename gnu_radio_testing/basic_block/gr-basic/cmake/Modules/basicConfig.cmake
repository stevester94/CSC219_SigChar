INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_BASIC basic)

FIND_PATH(
    BASIC_INCLUDE_DIRS
    NAMES basic/api.h
    HINTS $ENV{BASIC_DIR}/include
        ${PC_BASIC_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    BASIC_LIBRARIES
    NAMES gnuradio-basic
    HINTS $ENV{BASIC_DIR}/lib
        ${PC_BASIC_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(BASIC DEFAULT_MSG BASIC_LIBRARIES BASIC_INCLUDE_DIRS)
MARK_AS_ADVANCED(BASIC_LIBRARIES BASIC_INCLUDE_DIRS)

