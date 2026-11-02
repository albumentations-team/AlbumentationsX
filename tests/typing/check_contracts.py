import re
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
CASES = Path(__file__).resolve().parent


class TypeContracts(unittest.TestCase):
    def test_output_and_container_contracts(self) -> None:
        with TemporaryDirectory() as cache:
            for checker in ("mypy", "pyrefly"):
                for filename in ("positive.py", "negative.py"):
                    with self.subTest(checker=checker, fixture=filename):
                        source = CASES / filename
                        expected = {
                            number
                            for number, line in enumerate(source.read_text().splitlines(), start=1)
                            if line.endswith("# expected-error")
                        }
                        command = [sys.executable, "-m", checker]
                        if checker == "mypy":
                            command.extend(["--cache-dir", cache])
                        else:
                            command.extend(
                                [
                                    "check",
                                    "--python-interpreter-path",
                                    sys.executable,
                                    "--output-format",
                                    "min-text",
                                ]
                            )
                        # The interpreter and checked-in fixture paths are fixed by this test.
                        result = subprocess.run(  # noqa: S603
                            [*command, str(source)],
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        output = result.stdout + result.stderr
                        actual = {
                            int(match.group(1)) for match in re.finditer(re.escape(filename) + r":(\d+):", output)
                        }
                        assert result.returncode == (1 if expected else 0), output
                        assert actual == expected, output


if __name__ == "__main__":
    unittest.main()
