import ast
import unittest

from model_maven_worker.edit.source_ast import source_argument_insertions


class SourceArgumentInsertionTest(unittest.TestCase):
    def test_inserts_without_changing_existing_arguments_or_comments(self):
        cases = [
            "call()",
            "call(2)",
            "call(\n    # empty, but documented\n)",
            "call(\n    (2)  # existing, without trailing comma\n)",
            "call(\n    (2),  # existing trailing comma\n)",
            "call(\n    (2)\n    )",
            "call(\n    (2))",
            "call(\r\n    (2)  # keep, comment\r\n)",
        ]
        for expression in cases:
            with self.subTest(expression=expression):
                source = ("result = " + expression + "\n").encode()
                call = ast.parse(source).body[0].value
                edits = source_argument_insertions(source, call, [("offset", 3), ("scale", 4)])
                for start, end, replacement in sorted(edits, reverse=True):
                    source = source[:start] + replacement + source[end:]
                scope = {"call": lambda *args, **kwargs: (args, kwargs)}
                exec(compile(source, "fixture.py", "exec"), scope)
                expected_args = () if "empty" in expression or expression == "call()" else (2,)
                self.assertEqual(scope["result"], (expected_args, {"offset": 3, "scale": 4}))
                if "#" in expression:
                    self.assertIn(expression.split("#", 1)[1].splitlines()[0], source.decode())


if __name__ == "__main__":
    unittest.main()
