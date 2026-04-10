# =============================================================================
# Collect Source Files from c_api Directories
# =============================================================================

# c_api src files
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
set(C_API_SRC_FILES ${SRC_FILES} PARENT_SCOPE)

