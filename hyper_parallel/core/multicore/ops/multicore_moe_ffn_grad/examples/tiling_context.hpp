#ifndef MULTICORE_MOE_FFN_GRAD_TILING_CONTEXT_TEST_CODE
#define MULTICORE_MOE_FFN_GRAD_TILING_CONTEXT_TEST_CODE

struct TilingData {
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint64_t ubPartDataNum;
    uint64_t smallCoreTailDataNum;
    uint64_t bigCoreTailDataNum;
    uint64_t smallCoreLoopNum;
    uint64_t bigCoreLoopNum;
    uint64_t tailBlockNum;
};

struct SwiGluTilingData {
    uint32_t is32BAligned;
    uint32_t isDoubleBuffer;
    uint64_t rowLen;
    uint64_t colLen;
    uint32_t baseRowLen;
    uint32_t baseColLen;
    uint32_t activateLeft;
    uint32_t biasIsEmpty;
    uint32_t quantScaleIsEmpty;
    uint32_t activateScaleIsEmpty;
    uint64_t swiColLen;
    uint64_t perRowLen;
    uint64_t modRowLen;
    uint32_t usedCoreNum;
};

#endif // MULTICORE_MOE_FFN_GRAD_TILING_CONTEXT_TEST_CODE
