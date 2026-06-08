import json

from paths import (
    DEFAULT_PATH_ENTRIES,
    ensure_directories,
    get_paths,
    update_models_dir,
)


def test_get_paths_uses_relative_defaults(tmp_path):
    install_root = tmp_path / "app"
    install_root.mkdir()
    paths = get_paths(install_root=str(install_root), paths_file=str(tmp_path / "missing.json"))

    assert paths.models_dir == str(install_root / "data" / "models")
    assert paths.config_file == str(install_root / "data" / "automanager_config.json")
    assert paths.logs_dir == str(install_root / "logs")


def test_get_paths_loads_custom_absolute_paths(tmp_path):
    install_root = tmp_path / "app"
    install_root.mkdir()
    custom_models = tmp_path / "external-models"
    paths_file = install_root / "paths.json"
    paths_file.write_text(
        json.dumps(
            {
                **DEFAULT_PATH_ENTRIES,
                "models_dir": str(custom_models),
            }
        ),
        encoding="utf-8",
    )

    paths = get_paths(install_root=str(install_root), paths_file=str(paths_file))

    assert paths.models_dir == str(custom_models)


def test_ensure_directories_creates_expected_tree(tmp_path):
    install_root = tmp_path / "app"
    install_root.mkdir()
    paths_file = install_root / "paths.json"
    paths_file.write_text(json.dumps(DEFAULT_PATH_ENTRIES), encoding="utf-8")

    paths = ensure_directories(install_root=str(install_root), paths_file=str(paths_file))

    assert (install_root / "data" / "models").is_dir()
    assert (install_root / "data").is_dir()
    assert (install_root / "logs").is_dir()
    assert paths.manager_log.endswith("manager.log")


def test_update_models_dir_persists_and_creates_directory(tmp_path):
    install_root = tmp_path / "app"
    install_root.mkdir()
    paths_file = install_root / "paths.json"
    paths_file.write_text(json.dumps(DEFAULT_PATH_ENTRIES), encoding="utf-8")
    new_models = install_root / "external" / "models"

    paths = update_models_dir(
        "external/models",
        install_root=str(install_root),
        paths_file=str(paths_file),
    )

    assert paths.models_dir == str(new_models)
    assert new_models.is_dir()
    saved = json.loads(paths_file.read_text(encoding="utf-8"))
    assert saved["models_dir"] == "external/models"
