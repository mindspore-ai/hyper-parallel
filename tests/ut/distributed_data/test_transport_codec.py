# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for framed CPU payload serialization and integrity validation."""

import hashlib
import pickle
import unittest
from typing import Any

from hyper_parallel.distributed_data.schema import SampleKey
from hyper_parallel.distributed_data.transport import _decode_payload_segment, _encode_payload_segment


def _frame(value: Any) -> bytes:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return b"HPDDP1" + hashlib.sha256(payload).digest() + payload


class TestPayloadCodec(unittest.TestCase):
    """Verify route frames detect corruption and malformed sample identities."""

    def test_round_trip_preserves_order_keys_and_payloads(self) -> None:
        """A valid route segment round-trips without changing payload order."""
        items = (
            (SampleKey(0, 7), {"tokens": [1, 2], "image": b"jpeg"}),
            (SampleKey(4, 3), ("caption", 9)),
        )

        encoded = _encode_payload_segment(items)

        self.assertEqual(_decode_payload_segment(encoded), items)
        self.assertEqual(_encode_payload_segment(()), b"")
        self.assertEqual(_decode_payload_segment(b""), ())

    def test_encoder_rejects_duplicate_sample_keys(self) -> None:
        """One source route cannot claim the same raw sample twice."""
        duplicate = SampleKey(0, 1)

        with self.assertRaisesRegex(ValueError, "unique SampleKey"):
            _encode_payload_segment(((duplicate, "first"), (duplicate, "second")))

    def test_decoder_rejects_header_and_checksum_corruption(self) -> None:
        """Frame magic and SHA-256 digest are both validated before unpickling."""
        encoded = _encode_payload_segment(((SampleKey(0, 1), {"value": 3}),))
        bad_header = b"BROKEN" + encoded[6:]
        bad_payload = encoded[:-1] + bytes((encoded[-1] ^ 0x01,))

        with self.assertRaisesRegex(ValueError, "invalid frame header"):
            _decode_payload_segment(bad_header)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            _decode_payload_segment(bad_payload)

    def test_decoder_rejects_validly_framed_non_route_payload(self) -> None:
        """A matching digest does not bypass the route schema validation."""
        with self.assertRaisesRegex(ValueError, "invalid route segment"):
            _decode_payload_segment(_frame([(SampleKey(0, 0), "payload")]))

    def test_decoder_rejects_duplicate_keys_from_a_valid_pickle(self) -> None:
        """The receiver independently checks uniqueness instead of trusting the sender."""
        duplicate = SampleKey(1, 5)
        frame = _frame(((duplicate, "first"), (duplicate, "second")))

        with self.assertRaisesRegex(ValueError, "duplicate SampleKey"):
            _decode_payload_segment(frame)


if __name__ == "__main__":
    unittest.main()
