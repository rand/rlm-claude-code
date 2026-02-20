"""
Unit tests for response_parser module.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.response_parser import ResponseAction, ResponseParser


class TestResponseParser:
    """Tests for parser action extraction and compatibility behavior."""

    def test_parse_final_colon(self):
        parser = ResponseParser()
        parsed = parser.parse("Reasoning...\n\nFINAL: done")

        assert len(parsed) == 1
        assert parsed[0].action == ResponseAction.FINAL_ANSWER
        assert parsed[0].content == "done"
        assert "Reasoning" in parsed[0].reasoning

    def test_parse_final_call_form(self):
        parser = ResponseParser()
        parsed = parser.parse("FINAL(The result is 42)")

        assert len(parsed) == 1
        assert parsed[0].action == ResponseAction.FINAL_ANSWER
        assert parsed[0].content == "The result is 42"

    def test_parse_final_var_colon(self):
        parser = ResponseParser()
        parsed = parser.parse("FINAL_VAR: answer_var")

        assert len(parsed) == 1
        assert parsed[0].action == ResponseAction.FINAL_VAR
        assert parsed[0].content == "answer_var"

    def test_parse_final_var_call_form(self):
        parser = ResponseParser()
        parsed = parser.parse("FINAL_VAR(answer_var)")

        assert len(parsed) == 1
        assert parsed[0].action == ResponseAction.FINAL_VAR
        assert parsed[0].content == "answer_var"

    def test_parse_standalone_submit_as_repl_execute(self):
        parser = ResponseParser()
        parsed = parser.parse("SUBMIT({'answer': 'ok'})")

        assert len(parsed) == 1
        assert parsed[0].action == ResponseAction.REPL_EXECUTE
        assert "SUBMIT(" in parsed[0].content

    def test_parse_python_block_fallback_unchanged(self):
        parser = ResponseParser()
        parsed = parser.parse("Think first\n```python\nx = 1\nx\n```")

        assert len(parsed) == 1
        assert parsed[0].action == ResponseAction.REPL_EXECUTE
        assert parsed[0].content == "x = 1\nx"
        assert parsed[0].reasoning == "Think first"

    def test_has_final_answer_in_call_forms(self):
        parser = ResponseParser()

        assert parser.has_final_answer("FINAL(done)") is True
        assert parser.has_final_answer("FINAL_VAR(result)") is True
        assert parser.has_final_answer("No final yet") is False

    def test_extract_final_answer_in_call_forms(self):
        parser = ResponseParser()

        assert parser.extract_final_answer("FINAL(done)") == "done"
        assert parser.extract_final_answer("FINAL_VAR(result)") == "[Variable: result]"
