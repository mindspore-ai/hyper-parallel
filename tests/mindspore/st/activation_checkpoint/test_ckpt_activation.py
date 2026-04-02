#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""selective activation checkpoint test"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

BASE_SHARD = "ckpt_activation.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sac_group():
    """
    Feature: parallel run case in ckpt_activation
    Description:
        1. test_ac_memory_comparison
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(BASE_SHARD, "test_ac_memory_comparison", 11637, 1)
    ])
