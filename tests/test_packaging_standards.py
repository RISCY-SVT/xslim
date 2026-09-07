# Modified by RISCY-SVT in 2026: validate derivative license and CLI metadata.
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10
    from setuptools._vendor import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
SETUP_PATH = REPO_ROOT / "setup.py"
MANIFEST_PATH = REPO_ROOT / "MANIFEST.in"
SRC_PATH = REPO_ROOT / "src"
README_PATHS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "README_zh.md",
]
MAIN_MODULE_PATH = SRC_PATH / "xslim" / "__main__.py"


def _load_main_module():
    spec = importlib.util.spec_from_file_location(
        "xslim_cli_main", MAIN_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPackagingStandards(unittest.TestCase):
    def test_pyproject_declares_standard_build_metadata(self):
        pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            pyproject["build-system"]["build-backend"], "setuptools.build_meta"
        )
        project = pyproject["project"]
        self.assertEqual(project["dynamic"], ["version", "dependencies"])
        self.assertEqual(project["license"], "Apache-2.0")
        self.assertEqual(
            set(project["license-files"]),
            {
                "LICENSE",
                "LICENSE_AUDIT.md",
                "MODIFICATIONS.md",
                "NOTICE-RISCY-SVT",
                "THIRD_PARTY_NOTICES.md",
                "THIRD_PARTY_LICENSES.tsv",
                "UPSTREAM.md",
            },
        )
        self.assertEqual(project["scripts"]["xslim"], "xslim.__main__:main")
        self.assertEqual(project["authors"][0]["name"], "SpacemiT")
        self.assertEqual(project["maintainers"], [{"name": "RISCY-SVT"}])
        self.assertEqual(
            project["urls"]["Upstream"], "https://github.com/spacemit-com/xslim"
        )
        self.assertIn("riscy/k1x-yolo26", project["urls"]["Documentation"])
        self.assertEqual(
            project["scripts"]["xslim-yolo-output-check"],
            "xslim.tools.yolo_output_check:main",
        )
        self.assertEqual(
            project["scripts"]["xslim-qdq-boundary-audit"],
            "xslim.tools.qdq_boundary_audit:main",
        )
        self.assertEqual(
            project["scripts"]["xslim-spacemit-profile-check"],
            "xslim.tools.spacemit_profile:main",
        )
        self.assertEqual(pyproject["tool"]["setuptools"]["package-dir"], {"": "src"})
        self.assertEqual(
            pyproject["tool"]["setuptools"]["packages"]["find"]["where"],
            ["src"],
        )

    def test_setup_py_is_legacy_compatibility_shim(self):
        setup_text = SETUP_PATH.read_text(encoding="utf-8")

        self.assertIn('from setuptools import setup', setup_text)
        self.assertIn('setup()', setup_text)
        self.assertNotIn('install_requires', setup_text)
        self.assertNotIn('find_packages', setup_text)
        self.assertNotIn('version=', setup_text)

    def test_manifest_includes_repository_docs_and_samples(self):
        manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")

        self.assertIn('include README.md', manifest_text)
        self.assertIn('include README_zh.md', manifest_text)
        self.assertIn('graft doc', manifest_text)
        self.assertIn('graft docs', manifest_text)
        self.assertIn('graft samples', manifest_text)
        self.assertIn('prune samples/bert_quant_dataset', manifest_text)

    def test_readmes_document_standard_install_and_cli(self):
        for readme_path in README_PATHS:
            with self.subTest(readme=readme_path.name):
                readme_text = readme_path.read_text(encoding="utf-8")
                self.assertIn('python -m pip install .', readme_text)
                self.assertIn('python -m pip install -e .', readme_text)
                self.assertIn('python -m build', readme_text)
                self.assertIn('xslim --config config.json', readme_text)
                self.assertIn(
                    'python -m xslim --config config.json', readme_text
                )

    def test_cli_help_path_uses_standard_entry_name(self):
        main_module = _load_main_module()
        parser = main_module.build_parser()

        self.assertEqual(parser.prog, 'xslim')

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_module.main([])

        self.assertEqual(exit_code, 1)
        help_text = stdout.getvalue()
        self.assertIn('usage: xslim', help_text)
        self.assertIn('--config', help_text)
        self.assertIn('--input_path', help_text)
        self.assertIn('--output_path', help_text)

    def test_cli_version_reports_release_identity(self):
        main_module = _load_main_module()
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as exit_context:
            with redirect_stdout(stdout):
                main_module.main(["--version"])

        self.assertEqual(exit_context.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), "xslim 2.1.2+riscy.2.1")


if __name__ == '__main__':
    unittest.main()
