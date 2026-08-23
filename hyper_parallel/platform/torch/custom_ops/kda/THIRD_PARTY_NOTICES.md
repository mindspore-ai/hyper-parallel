# Third-Party Notices

The KDA Triton-Ascend kernels in the `triton` subdirectory are adapted from
the MIT-licensed `fla-org/flash-linear-attention` project. The common KDA
core was refreshed from revision
`27967b970eaaf982a6960abf6cba8add9c34c7cc`; Hyper-Parallel's state-summary
and context-parallel orchestration remain separate.

Source: <https://github.com/fla-org/flash-linear-attention>

The original copyright notices are retained in the adapted source files. The
applicable MIT license is distributed in the adjacent `LICENSE` file.

The Ascend Unified Buffer management helper is adapted through FLA from the
BSD-2-Clause-licensed `linkedin/Liger-Kernel` project.

Source: <https://github.com/linkedin/Liger-Kernel>

The applicable BSD 2-Clause license is distributed in the adjacent
`LICENSE-LIGER-BSD-2-CLAUSE` file.

Hyper-Parallel-specific orchestration and state-summary code remains licensed
under Apache License 2.0 as stated in those source files. Package-local import
adaptations do not remove or replace the original MIT notice.
