cmake_minimum_required(VERSION 3.24)
project(zynum VERSION 0.1.0 LANGUAGES C)
include(CTest)
if(NOT HETERIS_ABI_INCLUDE_DIR)
    find_path(HETERIS_ABI_INCLUDE_DIR heteris.h REQUIRED)
endif()
find_program(ZYNUM_ZIG zig REQUIRED)
execute_process(COMMAND "${ZYNUM_ZIG}" version OUTPUT_VARIABLE ZYNUM_ZIG_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT ZYNUM_ZIG_VERSION STREQUAL "0.17.0")
    message(FATAL_ERROR "The qualified toolchain is global Zig 0.17.0; found ${ZYNUM_ZIG_VERSION}.")
endif()
if(NOT TARGET heteris_data)
    add_library(heteris_data INTERFACE)
    target_include_directories(heteris_data INTERFACE "${HETERIS_ABI_INCLUDE_DIR}")
endif()
# Per-configuration prefixes and caches keep concurrent Debug/Release builds isolated.
set(ZYNUM_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/zig/$<CONFIG>")
set(ZYNUM_ARCHIVE "${ZYNUM_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}zynum${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(ZYNUM_OPTIMIZE "$<IF:$<CONFIG:Debug>,debug,safe>")
file(GLOB_RECURSE ZYNUM_ZIG_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.zig" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
list(APPEND ZYNUM_ZIG_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../src/scientific_platform_sources.zig"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/blas/types.zig"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/blas/core/shared/scalar.zig"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/numerics/recurrence_arithmetic.zig")
add_custom_command(OUTPUT "${ZYNUM_ARCHIVE}"
    COMMAND "${CMAKE_COMMAND}" -E env "ZIG_GLOBAL_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/zig-global-cache/$<CONFIG>"
            "${ZYNUM_ZIG}" build -Doptimize=${ZYNUM_OPTIMIZE}
            "-Dheteris-include=${HETERIS_ABI_INCLUDE_DIR}"
            --prefix "${ZYNUM_PREFIX}"
            --cache-dir "${CMAKE_CURRENT_BINARY_DIR}/zig-cache/$<CONFIG>"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS ${ZYNUM_ZIG_SOURCES} build.zig build.zig.zon include/zynum.h include/zynum_spectral.h
            "${HETERIS_ABI_INCLUDE_DIR}/heteris.h"
    VERBATIM)
add_custom_target(zynum_build DEPENDS "${ZYNUM_ARCHIVE}")
add_library(zynum STATIC IMPORTED GLOBAL)
set_target_properties(zynum PROPERTIES IMPORTED_LOCATION "${ZYNUM_ARCHIVE}")
# Imported locations do not evaluate generator expressions. Enumerate known configs.
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set_property(TARGET zynum PROPERTY IMPORTED_LOCATION_${config_upper}
        "${CMAKE_CURRENT_BINARY_DIR}/zig/${config}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}zynum${CMAKE_STATIC_LIBRARY_SUFFIX}")
endforeach()
if(NOT CMAKE_CONFIGURATION_TYPES)
    set_property(TARGET zynum PROPERTY IMPORTED_LOCATION
        "${CMAKE_CURRENT_BINARY_DIR}/zig/${CMAKE_BUILD_TYPE}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}zynum${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()
add_dependencies(zynum zynum_build)
target_include_directories(zynum INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(zynum INTERFACE heteris_data)
if(UNIX)
    target_link_libraries(zynum INTERFACE m)
endif()
if(BUILD_TESTING)
    add_executable(zynum_standalone tests/standalone.c)
    target_link_libraries(zynum_standalone PRIVATE zynum)
    target_compile_features(zynum_standalone PRIVATE c_std_11)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(zynum_standalone PRIVATE -Wall -Wextra -Werror -UNDEBUG)
    elseif(MSVC)
        target_compile_options(zynum_standalone PRIVATE /UNDEBUG)
    endif()
    add_test(NAME zynum_standalone COMMAND zynum_standalone)
    add_test(NAME zynum_zig_tests
        COMMAND "${CMAKE_COMMAND}" -E env "ZIG_GLOBAL_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/zig-global-cache/$<CONFIG>"
            "${ZYNUM_ZIG}" build test -Doptimize=${ZYNUM_OPTIMIZE}
            "-Dheteris-include=${HETERIS_ABI_INCLUDE_DIR}"
            --cache-dir "${CMAKE_CURRENT_BINARY_DIR}/zig-test-cache/$<CONFIG>")
    set_tests_properties(zynum_zig_tests PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 180)
endif()
install(FILES "${ZYNUM_ARCHIVE}" DESTINATION lib)
install(FILES include/zynum.h include/zynum_spectral.h DESTINATION include)
install(FILES "${HETERIS_ABI_INCLUDE_DIR}/heteris.h" DESTINATION include)
install(FILES ../../LICENSE ../../COPYING UPSTREAM.md DESTINATION share/licenses/zynum)
