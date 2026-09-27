import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "library") not in sys.path:
    sys.path.insert(0, str(ROOT / "library"))

import server
from llm.deepseek import DeepSeekProvider


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


class LLMTests(unittest.TestCase):
    def test_deepseek_json_request_and_usage(self):
        payload = {
            "model": "deepseek-flash",
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 3}},
        }
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(payload)) as urlopen:
            result = DeepSeekProvider(api_key="secret", model="deepseek-flash").generate(
                "return json", "question", json_output=True, temperature=0.0, max_tokens=64
            )
        request_body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual("deepseek-flash", request_body["model"])
        self.assertEqual({"type": "disabled"}, request_body["thinking"])
        self.assertEqual({"type": "json_object"}, request_body["response_format"])
        self.assertEqual((12, 4, 3), (result.input_tokens, result.output_tokens, result.cached_input_tokens))

    def test_query_understanding_is_combined_without_transcript(self):
        generated = mock.Mock()
        generated.value = {
            "original_question": "Registry 如何工作？",
            "english_query": "MCP Registry capability discovery",
            "keywords": ["server discovery"],
            "entities": ["MCP Registry"],
            "aliases": ["registry API"],
        }
        generated.usage.return_value = {"model": "deepseek-flash"}
        with mock.patch.object(server, "generate_json", return_value=generated) as call:
            value, usage = server.understand_query("Registry 如何工作？")
        self.assertEqual("deepseek-flash", usage["model"])
        self.assertEqual(
            "MCP Registry capability discovery server discovery MCP Registry registry API",
            server.retrieval_query(value),
        )
        self.assertEqual("Registry 如何工作？", call.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
