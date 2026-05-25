# pylint: skip-file
from paradise.common.layer_type import LayerType
from memory_estimation.estimate_v2 import EvaluatorV2
from memory_estimation.hooks.template import Template

# Instantiate evaluator with a model configuration,
#  log_level=0 removes warning messages
e = EvaluatorV2("./test_cases/mixtral/default.yaml", log_level=0)

# Check all defined node type
print(list(LayerType))

# Estimate peak memory (in Megabytes)
peak_mem = e.estimate_peak(verbose=True)
# Check whether estimation fits in device's max memory
e.mem_fit(peak_mem)

# Estimate static memory of a specific pipeline stage (in Megabytes)
print(e.static_mem_stage(1))
# Estimate dynamic memory of a specific pipeline stage (in Megabytes)
print(e.dynamic_mem_stage(1))
# Estimate static memory of a specific layer and stage (in Megabytes)
print(e.static_mem_layer(LayerType.FULL_REC_LAYER, 1))
# Estimate dynamic memory of a specific layer and stage (in Megabytes)
print(e.dynamic_mem_layer(LayerType.FULL_REC_LAYER, 1))
# Retrieve the memory estimation logs of a specific stage (in Megabytes)
print(e.logs_mem_stage(1))
# Fetch memory insights from each pipeline stage
stage_insights = e.estimate_peak_insight()
print(stage_insights)
# PPB Input
ppb_input = e.estimate_layer_memory()
print(ppb_input)

# Inspect a specific stage (here is the first one)
e.estimate_peak(spec_stage_id=0, verbose=True)

# Plot
e.estimate_peak(plot=True)

e = EvaluatorV2("./test_cases/deepseek3/default.yaml", log_level=0)


# Overwriting context function
def my_attn_num_param(ccfg, ctx):
    return 10 * ccfg.h * ccfg.h


e.set_attn_eval_fun(num_p=my_attn_num_param)

# Overwriting a training feature
e.set_passes(swap_os=True)


# Overwriting cost model variables
def custom(ccfg):
    ccfg.bytes_compute = 1
    ccfg.s = 1024
    ccfg.n_attMM = 5


e.set_ccfg(custom)

# Overwriting strategy
print(e.get_strategy())
e.set_strategy(dp=8, tp=8, m=128)
print(e.get_strategy())

e.estimate_peak(verbose=True)
# Inspect ccfg object (cost model variables)
e.print_ccfg()
# Inspect ctx object (evaluation variables and functions)
e.print_ctx()

# Load a hook class
# ... when declaring an Evaluator
e = EvaluatorV2(
    "./test_cases/deepseek3/default.yaml", log_level=0, hook_cls=Template()
)
# ... by using load_hook_cls()
e.load_hook_cls(Template())
