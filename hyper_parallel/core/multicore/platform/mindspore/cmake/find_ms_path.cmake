# =============================================================================
# Find MindSpore Internal Kernels Library
# =============================================================================

# Find Python to get MindSpore installation path
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Allow user to override MindSpore path
if(DEFINED ENV{MINDSPORE_PATH})
    set(MS_PATH $ENV{MINDSPORE_PATH})
    message(STATUS "Using MINDSPORE_PATH environment variable: ${MS_PATH}")
else()
    # Get MindSpore installation path using Python - get the last line of output
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import mindspore as ms; print(ms.__file__)"
        OUTPUT_VARIABLE MS_MODULE_PATH_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYTHON_RESULT
        ERROR_VARIABLE PYTHON_ERROR
    )

    # Extract the last non-empty line which should be the MindSpore path
    string(REPLACE "\n" ";" OUTPUT_LINES "${MS_MODULE_PATH_RAW}")

    # Find the last non-empty line
    set(MS_MODULE_PATH "")
    foreach(LINE ${OUTPUT_LINES})
        string(STRIP "${LINE}" STRIPPED_LINE)
        if(NOT STRIPPED_LINE STREQUAL "")
            set(MS_MODULE_PATH "${STRIPPED_LINE}")
        endif()
    endforeach()

    # Debug: Show the raw output and extracted path
    string(LENGTH "${MS_MODULE_PATH_RAW}" RAW_LENGTH)
    message(STATUS "Raw Python output length: ${RAW_LENGTH}")
    list(LENGTH OUTPUT_LINES NUM_LINES)
    message(STATUS "Number of output lines: ${NUM_LINES}")
    message(STATUS "Extracted MindSpore path: ${MS_MODULE_PATH}")

    # Validate the result
    if(NOT PYTHON_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to find MindSpore installation: ${PYTHON_ERROR}\n"
            "Please ensure MindSpore is installed, or set the MINDSPORE_PATH "
            "environment variable to the MindSpore installation directory.")
    endif()

    if(NOT MS_MODULE_PATH MATCHES ".*mindspore.*")
        message(FATAL_ERROR "Invalid MindSpore path detected: ${MS_MODULE_PATH}")
    endif()

    # Extract directory from MindSpore module path
    get_filename_component(MS_PATH ${MS_MODULE_PATH} DIRECTORY)
endif()

# =============================================================================
# MindSpore Path Detection
# =============================================================================

if(NOT DEFINED MS_PATH)
    message(FATAL_ERROR "MS_PATH is not defined. Make sure find_ms_path.cmake is included in the parent CMakeLists.txt")
endif()
