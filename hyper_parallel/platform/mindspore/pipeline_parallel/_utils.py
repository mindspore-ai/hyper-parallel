# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""pipeline parallel utils"""
import io
import pickle
import hyper_parallel
from mindspore import nn, Tensor, mint, ops
from mindspore.common import dtype as mstype
from mindspore.communication import GlobalComm
from mindspore.mint.distributed.distributed import _object_to_tensor, send, recv
from hyper_parallel.core.shard.local_func import custom_shard


class _MicroBatch(nn.Cell):
    """
    Split inputs into micro_batch in pipeline parallel.

    Args:
        micro_batch_num (int): The number of micro-batch.
        args_batch_dim (list, optional): Specify the batch dim of the args.
            Default ``None``.
        kwargs_batch_dim(dict, optional): Specify the batch dim of the kwargs.
            Default ``None``.
    Inputs:
        - **args** (list) - Input args.
        - **kwargs** (dict) - Input kwargs.

    Outputs:
        - **args_after_split** (list) - Input args after split into micro_batches.
        - **kwargs_after_split** (list) - Input kwargs after split into micro_batches.
    """

    def __init__(self, micro_batch_num, args_batch_dim=None, kwargs_batch_dim=None):
        super().__init__()
        self.micro_batch_num = micro_batch_num
        self.args_batch_dim = args_batch_dim
        self.kwargs_batch_dim = kwargs_batch_dim

    def construct(self, args, kwargs):
        """Construct of _MicroBatch"""
        args_after_split = []
        kwargs_after_split = []
        for micro_idx in range(self.micro_batch_num):
            micro_args = []
            micro_kwargs = {}
            for arg_idx, cur_arg in enumerate(args):
                cur_arg_batch_dim = 0
                if self.args_batch_dim and self.args_batch_dim[arg_idx] is not None:
                    cur_arg_batch_dim = self.args_batch_dim[arg_idx].batch_dim
                micro_arg = self.split_inputs_with_custom_shard(cur_arg, cur_arg_batch_dim, micro_idx)
                micro_args.append(micro_arg)
            args_after_split.append(micro_args)

            for key, cur_kwarg in kwargs.items():
                cur_kwarg_batch_dim = 0
                if self.kwargs_batch_dim is not None:
                    cur_kwarg_batch_dim = self.kwargs_batch_dim[key].batch_dim
                micro_kwarg = self.split_inputs_with_custom_shard(cur_kwarg, cur_kwarg_batch_dim, micro_idx)
                micro_kwargs[key] = micro_kwarg
            kwargs_after_split.append(micro_kwargs)
        return args_after_split, kwargs_after_split

    def split_inputs_with_custom_shard(self, input_tensor, cur_arg_batch_dim, micro_idx):
        if not isinstance(input_tensor, hyper_parallel.DTensor):
            raise TypeError(f"Input type {type(input_tensor)} is not DTensor.")
        input_layout = input_tensor.layout
        func_wrap = custom_shard(self.split_inputs, out_layouts=(input_layout,), in_layouts=(input_layout, None, None))
        return func_wrap(input_tensor, cur_arg_batch_dim, micro_idx)

    def split_inputs(self, input_tensor, cur_arg_batch_dim, micro_idx):
        """
        Split the input along the specified batch_dim and micro_idx
        """
        if cur_arg_batch_dim == -1:
            return input_tensor
        batch_dim_shape = input_tensor.shape[cur_arg_batch_dim]
        micro_batch_begin = (batch_dim_shape // self.micro_batch_num) * micro_idx
        micro_batch_end = (batch_dim_shape // self.micro_batch_num) * (micro_idx + 1)
        strided_slice_begin = [0] * input_tensor.ndim
        strided_slice_strides = [1] * input_tensor.ndim
        strided_slice_end = list(input_tensor.shape)
        strided_slice_begin[cur_arg_batch_dim] = micro_batch_begin
        strided_slice_end[cur_arg_batch_dim] = micro_batch_end
        micro_input = ops.strided_slice(input_tensor, strided_slice_begin, strided_slice_end, strided_slice_strides)
        return micro_input


def send_object_list(obj, dst=0, group=None):
    """
    Send the input Python object to dst rank.

    Args:
        obj (Any): The input tensor to be send.
        dst (int, optional): Specifies the global rank that send the Python object to.
            Default: ``0``.
        group (str, optional): Communication group. Default: ``None``.
    """
    if group is None:
        group = GlobalComm.WORLD_COMM_GROUP
    if not isinstance(group, str):
        raise TypeError(f"For 'send_object', the argument 'group' must be type of string, \
                          but got 'group' type : {type(group)}.")
    if not isinstance(dst, int):
        raise TypeError("For send_object, the dst must be int.")
    obj_tensor, tensor_size = _object_to_tensor(obj)
    obj_size = Tensor([tensor_size], dtype=mstype.int32)
    send(obj_size, dst, group)
    send(obj_tensor, dst, group)


def recv_object_list(recv_obj, src=0, group=None):
    """
    receive Python object from src rank.

    Args:
        recv_obj (list): list to recv python objects.
        src (int, optional): Specifies the global rank that receive the Python object.
            Default: ``0`` .
        group (str, optional): Communication group. Default: ``None``.
    """
    if group is None:
        group = GlobalComm.WORLD_COMM_GROUP
    if not isinstance(group, str):
        raise TypeError(f"For 'recv_object', the argument 'group' must be type of string, \
                          but got 'group' type : {type(group)}.")
    if not isinstance(src, int):
        raise TypeError("For recv_object, the src must be int.")
    obj_size = Tensor([0], dtype=mstype.int32)
    recv(obj_size, src, group)
    size_val = obj_size.item()
    obj_tensor = mint.empty([size_val], dtype=mstype.int8)
    recv(obj_tensor, src, group)
    buf = obj_tensor.asnumpy().tobytes()[:size_val]
    recv_obj.clear()
    recv_obj.append(pickle.Unpickler(io.BytesIO(buf)).load()[0])
