"""class ParaForNd inner use"""

import re
import toml
from fast_tuner.utils.common import cal_model_layers_num
from fast_tuner.utils.input_config import InputConfig
from fast_tuner.utils.logger import logger


class ParaForNd:
    """
    trans different params to inner param
    """
    def __init__(self, para):
        self.seq_len = 2048
        self.mf_args = None
        self.max_mem = 58 * 1024
        self.world_size = 1
        self.dp = 8 # dp_replicate_degree * dp_shard_degree(OP)
        self.dp_replicate_degree = 1
        self.dp_shard_degree = -1
        self.op = 1   # equal to real fsdp value
        self.pp = 1
        self.tp = 1
        self.cp = 1
        self.ep = 1
        self.etp = 1
        self.gbs = 8
        self.mbn = 1
        self.mbs = 1
        self.num_layers = 6
        self.swiglu = False
        self.qk_pos_emb_head_dim = 64
        self.qk_head_dim = 128
        self.q_lora_rank = None
        self.kv_lora_rank = None
        self.v_head_dim = 128
        self.expert_num = 8
        self.moe_intermediate_size = 256
        self.hidden_size = 256
        self.ffn_hidden_size = 1024
        self.num_attention_heads = 16
        self.padded_vocab_size = 2048
        self.first_k_dense_replace = 1
        self.use_distributed_optimizer = True
        self.kv_channels = 128
        self.group_query_attention = True
        self.num_query_groups = 8
        self.moe_shared_expert_intermediate_size = None
        self.multi_latent_attention = False
        # add for titan
        self.enable_fsdp_float8_all_gather = False
        self.precompute_float8_dynamic_scale_for_fsdp = False
        self.fsdp_reshard_after_forward = "default"
        self.enable_async_tensor_parallel = False
        self.pipeline_parallel_schedule = "Interleaved1F1B"
        self.model_name = None

        # for profile parser TODO: mindspeed only, should not be here
        self.num_layer_list = [4, 3]
        self.recompute_method = 'block'
        self.recompute_num_layers = 1
        self.profile_steps = 2
        self.get_args_from_file(para)

    def print_member_value(self):
        """Log all member attribute names and their current values."""
        for attr, value in self.__dict__.items():
            logger.info(f"{attr}: {value}")

    def convert_value(self, value_str):
        """Try to convert string to int/float, return original on failure"""
        if not value_str:
            return value_str
        # try int conversion (handle +/- integers)
        if value_str.lstrip('-').isdigit():
            return int(value_str)
        # try float conversion (handle +/- float, scientific notation)
        try:
            return float(value_str)
        except ValueError:
            return value_str  # return original if not numeric

    def parse_toml_parameters(self, toml_file_path):
        """Load and return parameters from a TOML configuration file."""
        params = toml.load(toml_file_path)
        return params

    # some vars not parsed, multi-line params and refs are parsed
    def parse_sh_parameters(self, sh_file_path):
        """
        Parse shell script params, return dict of module params.

        Args:
            sh_content: shell script content (string)
        Returns:
            dict: params by module, format {module_name: {param_name: param_value}}
        """
        # result: module_name -> {param_name: param_value}

        with open(sh_file_path, 'r', encoding='utf-8') as f:
            sh_content = f.read()
        result = {}

        # 1. Extract all vars (e.g. TP=1, DATA_PATH="xxx") for replacing refs like ${TP}
        variables = self.parse_variable(sh_content)

        # 2. Extract module params (e.g. GPT_ARGS, DATA_ARGS)
        self.parse_module_args(result, sh_content, variables)

        # 3. Flatten result: merge all module param dicts
        flattened_params = {}
        for _, params in result.items():
            # merge to global dict (later module overwrites on conflict)
            flattened_params.update(params)

        return flattened_params

    def parse_module_args(self, result, sh_content, variables):
        """
        Match module definition (e.g. GPT_ARGS="...params...")
        """
        module_pattern = re.compile(r'^(\w+)\s*=\s*"(.*?)"$', re.DOTALL | re.MULTILINE)
        for module_match in module_pattern.finditer(sh_content):
            module_name = module_match.group(1)  # e.g. GPT_ARGS or DATA_ARGS
            module_content = module_match.group(2)  # param content in module

            # parse params in module (--key value form)
            param_pattern = re.compile(r'--(\w[\w-]*)\s+([^\s\\]+)')
            # handle flag params (e.g. --use-flash-attn)
            flag_pattern = re.compile(r'--(\w[\w-]*)')

            module_params = {}

            # extract params with value first (e.g. --data-path $DATA_PATH)
            for match in param_pattern.finditer(module_content):
                param_key = match.group(1)  # param name: data-path
                param_value = match.group(2)  # param value: $DATA_PATH

                # replace var refs (e.g. $DATA_PATH -> actual value)
                for var_name, var_val in variables.items():
                    param_value = param_value.replace(f'${var_name}', var_val).replace(f'${{{var_name}}}', var_val)

                module_params[param_key.replace('-', '_')] = self.convert_value(param_value)

            # then extract flag params (e.g. --use-flash-attn)
            for match in flag_pattern.finditer(module_content):
                param_key = match.group(1)
                standard_key = param_key.replace('-', '_')
                if standard_key not in module_params:  # avoid overwriting value params
                    module_params[param_key.replace('-', '_')] = True  # True for flag on
            if module_params:
                result[module_name] = module_params

    def parse_variable(self, sh_content):
        """
        Extract all vars (e.g. TP=1, DATA_PATH="xxx") for replacing refs like ${TP}
        """
        variables = {}
        var_pattern = re.compile(r'^(\w+)\s*=\s*([\'"])(.*?)\2$', re.MULTILINE)
        for match in var_pattern.finditer(sh_content):
            var_name = match.group(1)  # var name (e.g. A)
            var_value = match.group(3)  # var value (e.g. cccc)
            variables[var_name] = var_value
        # handle unquoted vars (e.g. TP=1, PP=4)
        var_pattern2 = re.compile(r'^(\w+)\s*=\s*([\w.]+)$', re.MULTILINE)
        for match in var_pattern2.finditer(sh_content):
            var_name = match.group(1)
            var_value = match.group(2)
            variables[var_name] = var_value
        return variables

    def get_args_from_file(self, para):
        """
        get args from yaml/shell/toml file
        """
        if para.YAML_PATH:
            self.trans_yaml_param(para)

        elif para.SHELL_PATH:
            self.trans_shell_param(para)

        elif para.TOML_PATH:
            self.trans_toml_param(para)

        else:
            raise RuntimeError("only support yaml/shell/toml file, pls input valid config file")

        self.print_member_value()

    def trans_toml_param(self, para):
        """
        trans toml param to inner param
        """
        parsed_args = self.parse_toml_parameters(para.TOML_PATH)
        self.seq_len = parsed_args['training'].get('seq_len', 4096)
        self.world_size = para.NPUS_PER_NODE * para.NNODES
        self.pp = parsed_args['parallelism'].get('pipeline_parallel_degree', 1)
        self.tp = parsed_args['parallelism'].get('tensor_parallel_degree', 1)
        self.cp = parsed_args['parallelism'].get('context_parallel_degree', 1)
        self.dp_shard_degree = parsed_args['parallelism'].get('data_parallel_shard_degree', -1)
        if self.dp_shard_degree == -1:
            fsdp = self.world_size // self.pp // self.tp // self.cp // parsed_args['parallelism'].get(
                'data_parallel_replicate_degree', 1)
        else:
            fsdp = self.dp_shard_degree
        self.op = fsdp
        self.dp = parsed_args['parallelism'].get('data_parallel_replicate_degree', 1) * fsdp
        self.ep = parsed_args['parallelism'].get('expert_parallel_degree', 1)
        self.etp = parsed_args['parallelism'].get('expert_tensor_parallel_degree', 1)
        self.fsdp_reshard_after_forward = parsed_args['parallelism'].get('fsdp_reshard_after_forward', "default")
        self.enable_async_tensor_parallel = parsed_args['parallelism'].get('enable_async_tensor_parallel', False)
        local_batch_size = parsed_args['training'].get('local_batch_size', 1)
        self.gbs = local_batch_size * self.dp
        self.mbs = parsed_args['parallelism'].get('pipeline_parallel_microbatch_size', 1)
        self.mbn = local_batch_size // self.mbs
        self.enable_fsdp_float8_all_gather = parsed_args['quantize']['linear']['float8'].get(
            'enable_fsdp_float8_all_gather', False)
        self.precompute_float8_dynamic_scale_for_fsdp = parsed_args['quantize']['linear']['float8'].get(
            'precompute_float8_dynamic_scale_for_fsdp', False)
        self.model_name = parsed_args['model'].get('name', "llama3")
        model_flavor = parsed_args['model'].get('flavor', "8B")
        try:
            # pylint: disable=C0415
            import torchtitan.protocols.train_spec as train_spec_module
            train_spec = train_spec_module.get_train_spec(self.model_name)
            model_args = train_spec.model_args[model_flavor]

            self.num_layers = model_args.n_layers
            self.qk_pos_emb_head_dim = getattr(model_args, 'qk_rope_head_dim', 64)
            self.qk_head_dim = getattr(model_args, 'qk_nope_head_dim', 128)
            self.q_lora_rank = getattr(model_args, 'q_lora_rank', None)
            self.kv_lora_rank = getattr(model_args, 'kv_lora_rank', None)
            self.v_head_dim = getattr(model_args, 'v_head_dim', None)
            self.expert_num = getattr(getattr(model_args, 'moe_args', None), 'num_experts', None)
            self.moe_intermediate_size = getattr(model_args, 'moe_inter_dim', None)
            self.hidden_size = getattr(model_args, 'dim', None)
            self.ffn_hidden_size = getattr(model_args, 'inter_dim', 4 * self.hidden_size)
            self.cal_ffn_hidden_size(model_args)
            self.num_attention_heads = getattr(model_args, 'n_heads', None)
            self.padded_vocab_size = getattr(model_args, 'vocab_size', None)
            self.first_k_dense_replace = getattr(model_args, 'n_dense_layers', 0)
            self.use_distributed_optimizer = self.dp_shard_degree != 1
            self.kv_channels = self.hidden_size // self.num_attention_heads
            n_kv_heads = getattr(model_args, 'n_kv_heads', None)
            self.group_query_attention = (n_kv_heads is not None and n_kv_heads != self.num_attention_heads)
            self.num_query_groups = n_kv_heads
            self.moe_shared_expert_intermediate_size = self.moe_intermediate_size * getattr(
                getattr(model_args, 'moe_args', None), 'num_shared_experts', 0)
            self.multi_latent_attention = bool(self.q_lora_rank)

        except (ImportError, AttributeError) as e:
            print(f'Error is: {e}')
        self.profile_steps = 1

    def trans_yaml_param(self, para):
        """
        trans yaml param to inner param
        """
        input_args = InputConfig(para.YAML_PATH)
        self.dp = input_args.parallel_config.data_parallel
        self.tp = input_args.parallel_config.model_parallel
        self.pp = input_args.parallel_config.pipeline_stage
        self.cp = 1 if input_args.parallel_config.get('context_parallel') is None else input_args.parallel_config.get(
            'context_parallel')
        self.world_size = self.dp * self.tp * self.cp * self.pp
        self.num_layers = cal_model_layers_num(input_args)
        self.mbn = input_args.parallel_config.micro_batch_num
        self.max_mem = input_args.context.max_device_memory
        self.expert_num = None if input_args.moe_config is None else input_args.context.expert_num
        self.mf_args = input_args

    def cal_ffn_hidden_size(self, model_args):
        """
        Update ffn_hidden_size
        """
        print(f"origin_ffn_hidden_size {self.ffn_hidden_size}")
        multiple_of = getattr(model_args, 'multiple_of', None)
        if multiple_of is None:
            return
        ffn_dim_multiplier = getattr(model_args, 'ffn_dim_multiplier')
        if multiple_of and ffn_dim_multiplier:
            tmp_ffn_hidden_size = int(2 * self.ffn_hidden_size / 3)
            tmp_ffn_hidden_size = int(ffn_dim_multiplier * tmp_ffn_hidden_size)
            self.ffn_hidden_size = multiple_of * ((tmp_ffn_hidden_size + multiple_of - 1) // multiple_of)
        print("current_ffn_hidden_size {self.ffn_hidden_size}")

    def trans_shell_param(self, para):
        """
        trans shell param to inner param
        """
        parsed_args = self.parse_sh_parameters(para.SHELL_PATH)
        self.world_size = parsed_args['nproc_per_node'] * parsed_args['nnodes']
        self.gbs = parsed_args.get('global_batch_size')
        self.mbs = parsed_args.get('micro_batch_size')
        self.pp = parsed_args.get('pipeline_model_parallel_size', 1)
        self.tp = parsed_args.get('tensor_model_parallel_size', 1)
        self.cp = parsed_args.get('context_parallel_size', 1)
        self.dp = self.world_size // self.pp // self.tp // self.cp
        self.mbn = self.gbs // self.mbs // self.dp
        self.num_layers = parsed_args.get('num_layers')
        self.swiglu = parsed_args.get('swiglu', False)
        self.qk_pos_emb_head_dim = parsed_args.get('qk_pos_emb_head_dim', 64)
        self.qk_head_dim = parsed_args.get('qk_head_dim', 128)
        self.q_lora_rank = parsed_args.get('q_lora_rank')
        self.kv_lora_rank = parsed_args.get('kv_lora_rank', 32)
        self.v_head_dim = parsed_args.get('v_head_dim', 128)
        self.expert_num = parsed_args.get('num_experts')
        self.hidden_size = parsed_args.get('hidden_size')
        self.ffn_hidden_size = parsed_args.get('ffn_hidden_size', 4 * self.hidden_size)
        self.moe_intermediate_size = parsed_args.get('moe_intermediate_size', self.ffn_hidden_size)
        self.num_attention_heads = parsed_args.get('num_attention_heads')
        self.padded_vocab_size = parsed_args.get('padded_vocab_size', parsed_args.get('vocab_size'))
        self.first_k_dense_replace = parsed_args.get('first_k_dense_replace', 0)
        self.use_distributed_optimizer = parsed_args.get('use_distributed_optimizer', False)
        self.kv_channels = parsed_args.get('kv_channels', self.hidden_size // self.num_attention_heads)
        self.group_query_attention = parsed_args.get('group_query_attention', False)
        self.moe_shared_expert_intermediate_size = parsed_args.get('moe_shared_expert_intermediate_size')
        self.multi_latent_attention = parsed_args.get('multi_latent_attention', False)
