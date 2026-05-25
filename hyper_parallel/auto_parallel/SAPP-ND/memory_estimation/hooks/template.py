# pylint: skip-file
from memory_estimation.hook_base import MemEvalHook, hook_runner


class Template(MemEvalHook):

    # Formula to get hooked
    @staticmethod
    def f(ccfg, ctx):
        return None

    # Overwriting cost model variable
    @staticmethod
    def custom_ccfg(ccfg):
        pass

    @staticmethod
    @hook_runner("model name")
    def run_hooks(e):
        c = Template
        e.set_ccfg(c.custom_ccfg)
        e.set_passes(vpp_less_mem=False, swap_os=False, dropless_tok_factor=1)
        e.set_head_eval_fun(num_p=c.f, stat=c.f, dyn=c.f)
        e.set_tail_eval_fun(num_p=c.f, stat=c.f, dyn=c.f)
        e.set_body_eval_fun(
            "NOT_REC_LAYER",
            num_p=c.f,
            stat_p=c.f,
            stat_os=c.f,
            stat_grad=c.f,
            dyn_activ=c.f,
            dyn_dp_comm=c.f,
            dyn_tp_comm=c.f,
            dyn_cp_comm=c.f,
            dyn_ep_comm=c.f,
        )
        e.set_attn_eval_fun(
            num_p=c.f,
            qkv=c.f,
            score=c.f,
            proj=c.f,
        )
        e.set_ffn_eval_fun(
            num_p=c.f,
            activation=c.f,
            moe_activ=c.f
        )
        e.set_norm_eval_fun(num_p=c.f, activation=c.f)
        e.set_pp_micro_factor_eval_fun("1f1b", c.f)
