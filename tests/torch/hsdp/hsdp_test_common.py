#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""hsdp test common"""
import time
from torch import optim
from hyper_parallel import hsdp_wait_grad_handle


def train(model, data, comm_async=True, train_steps=10):
    """train net with data"""
    train_steps = max(train_steps, 1)
    train_steps = train_steps + 1

    model = model.npu()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    cost_time = 0
    for i in range(train_steps):
        if i != 0:
            start_time = time.time()
        loss = model(data)
        loss.backward()
        if comm_async:
            hsdp_wait_grad_handle()
        if i != 0:
            end_time = time.time()
            cost_time = end_time - start_time + cost_time
        optimizer.step()
        optimizer.zero_grad()
    return cost_time
