# Third-Party Notices

The Gated DeltaNet Triton implementation in this directory is adapted from
MindSpeed-MM's `mindspeed_mm/fsdp/ops/gdn` implementation. The copied source
files retain their original copyright headers. See `LICENSE` in this directory
for the applicable redistribution terms; files with an explicit license header
continue to be governed by that header.

Hyper-Parallel changes remove MindSpeed-MM training-context and activation-
offload dependencies, use package-local imports, and expose the implementation
through Hyper-Parallel's Torch custom-op interface.
