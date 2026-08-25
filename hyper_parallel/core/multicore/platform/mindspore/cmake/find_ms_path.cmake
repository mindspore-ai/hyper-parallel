# =============================================================================
# Find MindSpore without importing the framework or modifying the environment.
# =============================================================================
find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(DEFINED ENV{MINDSPORE_PATH})
    set(MS_PATH $ENV{MINDSPORE_PATH})
    set(MS_VERSION "user-specified")
else()
    set(MS_DISCOVERY_SCRIPT [=[
import importlib.metadata as metadata
import importlib.util
import sys

spec = importlib.util.find_spec("mindspore")
if spec is None or not spec.submodule_search_locations:
    sys.exit(1)
print(next(iter(spec.submodule_search_locations)) + "|" + metadata.version("mindspore"))
]=])
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c ${MS_DISCOVERY_SCRIPT}
        OUTPUT_VARIABLE MS_DISCOVERY
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE MS_DISCOVERY_ERROR
        RESULT_VARIABLE MS_DISCOVERY_RESULT
    )
    if(NOT MS_DISCOVERY_RESULT EQUAL 0)
        message(FATAL_ERROR
            "MindSpore is not installed for ${Python3_EXECUTABLE}. "
            "Install the declared build dependency or set MINDSPORE_PATH. "
            "Discovery error: ${MS_DISCOVERY_ERROR}")
    endif()
    string(REPLACE "|" ";" MS_DISCOVERY_FIELDS "${MS_DISCOVERY}")
    list(LENGTH MS_DISCOVERY_FIELDS MS_DISCOVERY_FIELD_COUNT)
    if(NOT MS_DISCOVERY_FIELD_COUNT EQUAL 2)
        message(FATAL_ERROR "Unexpected MindSpore discovery result: ${MS_DISCOVERY}")
    endif()
    list(GET MS_DISCOVERY_FIELDS 0 MS_PATH)
    list(GET MS_DISCOVERY_FIELDS 1 MS_VERSION)
endif()

if(NOT IS_DIRECTORY "${MS_PATH}" OR NOT EXISTS "${MS_PATH}/include")
    message(FATAL_ERROR "Invalid MindSpore development path: ${MS_PATH}")
endif()
message(STATUS "MindSpore path: ${MS_PATH}")
message(STATUS "MindSpore version: ${MS_VERSION}")
