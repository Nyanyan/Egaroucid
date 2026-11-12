"""Check exported PDFs survive Git checkout with automatic CRLF conversion.

Usage: python src/tools/diagnostics/test_board_image_pdf_git.py PDF_DIRECTORY
The directory can contain samples from test_board_image_vector or GUI exports.
"""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    samples = sorted(args.directory.glob("*.pdf"))
    if not samples:
        parser.error("No PDF samples found")
    build = (ROOT / "build/pdf_git_tests").resolve()
    build.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=build) as directory:
        repo = Path(directory).resolve()
        if not repo.is_relative_to(build):
            raise RuntimeError("Temporary Git repository is outside the test build directory")
        environment = dict(os.environ, GIT_ATTR_NOSYSTEM="1")

        def git(*arguments):
            return subprocess.run(["git", "-C", str(repo), *arguments], env=environment,
                                  check=True, capture_output=True, text=True).stdout

        git("init", "-q")
        git("config", "core.autocrlf", "true")
        git("config", "core.safecrlf", "false")
        attributes = repo / "empty-attributes"
        attributes.write_bytes(b"")
        git("config", "core.attributesFile", str(attributes))
        originals = {sample.name: sample.read_bytes() for sample in samples}
        for name, data in originals.items():
            (repo / name).write_bytes(data)
        git("add", "--", *originals)
        errors = []
        for line in git("ls-files", "--eol").splitlines():
            if line.split()[:2] != ["i/-text", "w/-text"]:
                errors.append("Git did not detect a binary PDF: " + line)
        for name in originals:
            (repo / name).unlink()
        git("checkout-index", "--all")
        for name, expected in originals.items():
            if (repo / name).read_bytes() != expected:
                errors.append("Git checkout changed the PDF bytes: " + name)
        if errors:
            print("\n".join(errors))
            return 1
    print(f"PASS: {len(samples)} PDFs detected as binary and unchanged by Git checkout (core.autocrlf=true)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
