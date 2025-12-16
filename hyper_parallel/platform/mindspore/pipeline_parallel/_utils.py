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
import mindspore as ms
from hyper_parallel.core.dtensor import DTensor
from mindspore import nn, Tensor, mint, ops
from mindspore.common import dtype as mstype
from mindspore.mint.distributed.distributed import _object_to_tensor, send, recv
from hyper_parallel.core.tensor_parallel.local_func import custom_shard
from mindspore.communication import GlobalComm


class BatchDimSpec:
    """
    Specify the batch dimension of a Tensor.

    Args:
        batch_dim(int): batch dimension。
    """
    __slots__ = ("batch_dim",)

    def __init__(self, batch_dim):
        if not isinstance(batch_dim, int):
            raise TypeError(f"batch_dim must be int, but got type {type(batch_dim)}.")
        self.batch_dim = batch_dim

    def __repr__(self):
        return f"BatchDimSpec({self.batch_dim})"

    def __str__(self):
        return f"BatchDim(dim={self.batch_dim})"

    @staticmethod
    def from_tuple(batch_dims):
        if not isinstance(batch_dims, tuple):
            raise TypeError(f"batch_dims must be tuple, but got type {type(batch_dims)}.")
        return tuple(BatchDimSpec(dim) for dim in batch_dims)

    @staticmethod
    def from_dict(batch_dims):
        if not isinstance(batch_dims, dict):
            raise TypeError(f"batch_dims must be dict, but got type {type(batch_dims)}.")
        return {k: BatchDimSpec(v) for k, v in batch_dims.items()}

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

    def split_inputs_with_custom_shard(self, input, cur_arg_batch_dim, micro_idx):
        # if not isinstance(input, ms.parallel.DTensor):
        if not isinstance(input, mindspore-parallel.hyper_parallel.core.dtensor.DTensor):
            raise TypeError(f"Input type {type(input)} is not DTensor.")
        input_layout = input.layout
        func_wrap = custom_shard(self.split_inputs, out_layouts=(input_layout,), in_layouts=(input_layout, None, None))
        return func_wrap(input, cur_arg_batch_dim, micro_idx)

    def split_inputs(self, input, cur_arg_batch_dim, micro_idx):
        """
        Split the input along the specified batch_dim and micro_idx
        """
        if cur_arg_batch_dim == -1:
            return input
        batch_dim_shape = input.shape[cur_arg_batch_dim]
        micro_batch_begin = (batch_dim_shape // self.micro_batch_num) * micro_idx
        micro_batch_end = (batch_dim_shape // self.micro_batch_num) * (micro_idx + 1)
        strided_slice_begin = [0] * input.ndim
        strided_slice_strides = [1] * input.ndim
        strided_slice_end = list(input.shape)
        strided_slice_begin[cur_arg_batch_dim] = micro_batch_begin
        strided_slice_end[cur_arg_batch_dim] = micro_batch_end
        micro_input = ops.strided_slice(input, strided_slice_begin, strided_slice_end, strided_slice_strides)
        return micro_input

class _RecvInfo:
    """
    Used for construct forward Receive operation and backward Send operation.
    """

    def __init__(self, global_rank, buffer=None):
        self._global_rank = global_rank
        self._buffer = buffer

    @property
    def global_rank(self):
        return self._global_rank

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, val):
        self._buffer = val


def send_object(obj, dst=0, group=None):
    """
    send the input Python object to dst rank.

    Note:
        - Similar to :func:`mindspore.mint.distributed.send`, but Python objects can be passed in.
        - Only support PyNative mode, Graph mode is not currently supported.

    Args:
        object (Any): The input to be send.
        dst (int, optional): Specifies the rank(global rank) of the process that send the Python object to.
            Default: ``0`` .
        group (str, optional): The communication group to work on. If ``None``, which means ``"hccl_world_group"`` in
            Ascend. Default: ``None``.

    Raises:
        TypeError: If `dst` is not an integer or `group` is not a string.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend``
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

def recv_object(src=0, group=None):
    """
    receive Python object from src rank.

    Note:
        - Similar to :func:`mindspore.mint.distributed.recv`, but Python objects can be received.
        - Only support PyNative mode, Graph mode is not currently supported.

    Args:
        src (int, optional): Specifies the rank(global rank) of the process that receive the Python object.
            Default: ``0`` .
        group (str, optional): The communication group to work on. If ``None``, which means ``"hccl_world_group"`` in
            Ascend. Default: ``None``.

    Raises:
        TypeError: If `src` is not an integer or `group` is not a string.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend``
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
    return pickle.Unpickler(io.BytesIO(buf)).load()