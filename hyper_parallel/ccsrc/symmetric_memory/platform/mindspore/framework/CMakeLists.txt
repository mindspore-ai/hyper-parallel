# =============================================================================
# Base Source Files Collection
# =============================================================================

# Collect all .cc files recursively from the base directory
file(GLOB_RECURSE BASE_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

# Make BASE_SRC_FILES available to parent scope
set(FRAMEWORK_SRC_FILES ${BASE_SRC_FILES} PARENT_SCOPE)
