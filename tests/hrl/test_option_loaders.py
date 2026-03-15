"""Unit tests for OptionLibraryLoader class."""

import pytest
import yaml
from pathlib import Path

from abx_amr_simulator.hrl import OptionLibraryLoader
# Import test helpers (sys.path configured in tests/conftest.py)
from test_reference_helpers import create_mock_environment  # type: ignore[import-not-found]

def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding='utf-8')


class TestOptionLibraryLoaderBasic:
    """Basic tests for OptionLibraryLoader."""

    def test_load_library_from_yaml(self, tmp_path):
        """Test loading library from YAML config."""
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()

        default_config = {'antibiotic': 'A', 'duration': 5}
        default_config_path = tmp_path / 'block_option_default_config.yaml'
        _write_yaml(path=default_config_path, data=default_config)

        lib_config = {
            'library_name': 'test_lib',
            'description': 'Test library',
            'options': [
                {
                    'option_name': 'A_5',
                    'option_type': 'block',
                    'option_subconfig_file': str(default_config_path),
                    'config_params_override': {'antibiotic': 'A', 'duration': 5},
                }
            ]
        }
        lib_config_path = lib_dir / 'test_lib.yaml'
        _write_yaml(path=lib_config_path, data=lib_config)

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib, resolved = OptionLibraryLoader.load_library(
            library_config_path=str(lib_config_path),
            env=env,
        )

        assert lib.name == 'test_lib'
        assert len(lib) == 1
        assert 'A_5' in lib.options
        assert resolved['library_name'] == 'test_lib'
        assert len(resolved['options']) == 1

    def test_load_library_from_yaml_canonical_alternation(self, tmp_path):
        """Test canonical alternation option loads via built-in loader mapping."""
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()

        default_config = {'sequence': ['A', 'B', 'A']}
        default_config_path = tmp_path / 'alternation_option_default_config.yaml'
        _write_yaml(path=default_config_path, data=default_config)

        lib_config = {
            'library_name': 'test_alternation_lib',
            'options': [
                {
                    'option_name': 'ALT_A_B',
                    'option_type': 'alternation',
                    'option_subconfig_file': str(default_config_path),
                }
            ]
        }
        lib_config_path = lib_dir / 'test_alternation_lib.yaml'
        _write_yaml(path=lib_config_path, data=lib_config)

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib, resolved = OptionLibraryLoader.load_library(
            library_config_path=str(lib_config_path),
            env=env,
        )

        assert len(lib) == 1
        assert 'ALT_A_B' in lib.options
        assert lib['ALT_A_B'].k == 3
        assert resolved['options'][0]['option_type'] == 'alternation'
        assert resolved['options'][0]['merged_config']['sequence'] == ['A', 'B', 'A']

    def test_load_library_from_yaml_canonical_heuristic(self, tmp_path):
        """Test canonical heuristic option loads via built-in loader mapping."""
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()

        default_config = {
            'duration': 7,
            'action_thresholds': {
                'prescribe_A': 0.5,
                'prescribe_B': 0.5,
                'no_treatment': 0.0,
            },
            'uncertainty_threshold': 2.0,
        }
        default_config_path = tmp_path / 'heuristic_option_default_config.yaml'
        _write_yaml(path=default_config_path, data=default_config)

        lib_config = {
            'library_name': 'test_heuristic_lib',
            'options': [
                {
                    'option_name': 'HEURISTIC_BASE',
                    'option_type': 'heuristic',
                    'option_subconfig_file': str(default_config_path),
                }
            ]
        }
        lib_config_path = lib_dir / 'test_heuristic_lib.yaml'
        _write_yaml(path=lib_config_path, data=lib_config)

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib, resolved = OptionLibraryLoader.load_library(
            library_config_path=str(lib_config_path),
            env=env,
        )

        assert len(lib) == 1
        assert 'HEURISTIC_BASE' in lib.options
        assert lib['HEURISTIC_BASE'].k == 7
        assert resolved['options'][0]['option_type'] == 'heuristic'
        assert resolved['options'][0]['merged_config']['duration'] == 7

    def test_load_library_missing_file(self):
        """Test loading from nonexistent file."""
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        with pytest.raises(FileNotFoundError):
            OptionLibraryLoader.load_library(library_config_path='/nonexistent/path/config.yaml', env=env)

    def test_load_library_empty_config(self, tmp_path):
        """Test loading from empty config."""
        lib_config_path = tmp_path / 'empty.yaml'
        lib_config_path.write_text('', encoding='utf-8')
        
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        with pytest.raises(ValueError):
            OptionLibraryLoader.load_library(library_config_path=str(lib_config_path), env=env)

    def test_load_library_no_options(self, tmp_path):
        """Test loading config with no options."""
        lib_config = {
            'library_name': 'empty_lib',
            'options': []
        }
        lib_config_path = tmp_path / 'empty_lib.yaml'
        lib_config_path.write_text(yaml.dump(lib_config), encoding='utf-8')
        
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        with pytest.raises(ValueError):
            OptionLibraryLoader.load_library(library_config_path=str(lib_config_path), env=env)


class TestOptionLibraryLoaderMultipleOptions:
    """Test loading multiple options."""

    def test_load_multiple_options(self, tmp_path):
        """Test loading library with multiple options."""
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()

        default_config = {'antibiotic': 'A', 'duration': 5}
        default_config_path = tmp_path / 'block_option_default_config.yaml'
        _write_yaml(path=default_config_path, data=default_config)

        lib_config = {
            'library_name': 'test_lib',
            'options': [
                {
                    'option_name': f'OPT_{i}',
                    'option_type': 'block',
                    'option_subconfig_file': str(default_config_path),
                    'config_params_override': {'antibiotic': 'A', 'duration': i*5},
                }
                for i in range(1, 4)
            ]
        }
        lib_config_path = lib_dir / 'test_lib.yaml'
        _write_yaml(path=lib_config_path, data=lib_config)

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        lib, _ = OptionLibraryLoader.load_library(
            library_config_path=str(lib_config_path),
            env=env,
        )

        assert len(lib) == 3
        assert lib['OPT_1'].k == 5
        assert lib['OPT_2'].k == 10
        assert lib['OPT_3'].k == 15


class TestOptionLibraryLoaderErrors:
    """Test error handling in loader."""

    def test_loader_missing_option_name(self, tmp_path):
        """Test error when option missing 'option_name'."""
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()
        
        lib_config = {
            'library_name': 'test_lib',
            'options': [
                {
                    'option_type': 'block',
                    # Missing 'option_name'
                }
            ]
        }
        lib_config_path = lib_dir / 'test_lib.yaml'
        _write_yaml(path=lib_config_path, data=lib_config)
        
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        with pytest.raises(RuntimeError) as exc_info:
            OptionLibraryLoader.load_library(library_config_path=str(lib_config_path), env=env)
        
        assert 'option_name' in str(exc_info.value)

    def test_loader_missing_subconfig_file(self, tmp_path):
        """Test error when option subconfig file not found."""
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()
        
        lib_config = {
            'library_name': 'test_lib',
            'options': [
                {
                    'option_name': 'OPT_1',
                    'option_type': 'block',
                    'option_subconfig_file': 'nonexistent/config.yaml',
                }
            ]
        }
        lib_config_path = lib_dir / 'test_lib.yaml'
        _write_yaml(path=lib_config_path, data=lib_config)
        
        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        with pytest.raises(RuntimeError):
            OptionLibraryLoader.load_library(library_config_path=str(lib_config_path), env=env)

    def test_unsupported_option_type_fails_loudly(self, tmp_path):
        subconfig_path = tmp_path / 'x.yaml'
        _write_yaml(path=subconfig_path, data={'antibiotic': 'A', 'duration': 5})

        lib_config = {
            'library_name': 'test_lib',
            'options': [
                {
                    'option_name': 'BAD',
                    'option_type': 'Block',
                    'option_subconfig_file': str(subconfig_path),
                }
            ]
        }
        config_path = tmp_path / 'lib.yaml'
        _write_yaml(path=config_path, data=lib_config)

        env = create_mock_environment(antibiotic_names=['A'])
        with pytest.raises(RuntimeError, match="unsupported option_type 'Block'.*Allowed values are"):
            OptionLibraryLoader.load_library(
                library_config_path=str(config_path),
                env=env,
            )

    def test_canonical_with_plugin_fields_fails_loudly(self, tmp_path):
        subconfig_path = tmp_path / 'x.yaml'
        _write_yaml(path=subconfig_path, data={'antibiotic': 'A', 'duration': 5})

        lib_config = {
            'library_name': 'test_lib',
            'options': [
                {
                    'option_name': 'BAD',
                    'option_type': 'block',
                    'option_subconfig_file': str(subconfig_path),
                    'plugin': {
                        'loader_module': 'x.py',
                        'loader_function': 'load_custom_option',
                    },
                }
            ]
        }
        config_path = tmp_path / 'lib.yaml'
        _write_yaml(path=config_path, data=lib_config)

        env = create_mock_environment(antibiotic_names=['A'])
        with pytest.raises(RuntimeError, match="canonical option_type 'block'.*must not include plugin"):
            OptionLibraryLoader.load_library(
                library_config_path=str(config_path),
                env=env,
            )

    @pytest.mark.parametrize('missing_key', ['loader_module', 'loader_function'])
    def test_custom_missing_plugin_fields_fail_loudly(self, tmp_path, missing_key):
        subconfig_path = tmp_path / 'x.yaml'
        _write_yaml(path=subconfig_path, data={'duration': 5})

        plugin_config = {
            'loader_module': 'x.py',
            'loader_function': 'load_custom_option',
        }
        plugin_config.pop(missing_key)

        lib_config = {
            'library_name': 'test_lib',
            'options': [
                {
                    'option_name': 'BAD_CUSTOM',
                    'option_type': 'custom',
                    'option_subconfig_file': str(subconfig_path),
                    'plugin': plugin_config,
                }
            ]
        }
        config_path = tmp_path / 'lib.yaml'
        _write_yaml(path=config_path, data=lib_config)

        env = create_mock_environment(antibiotic_names=['A'])
        with pytest.raises(RuntimeError, match='missing required plugin fields'):
            OptionLibraryLoader.load_library(
                library_config_path=str(config_path),
                env=env,
            )

    @pytest.mark.parametrize('legacy_key', ['loader_module', 'loader_function'])
    def test_legacy_top_level_loader_keys_rejected(self, tmp_path, legacy_key):
        subconfig_path = tmp_path / 'x.yaml'
        _write_yaml(path=subconfig_path, data={'antibiotic': 'A', 'duration': 5})

        option_spec = {
            'option_name': 'BAD',
            'option_type': 'block',
            'option_subconfig_file': str(subconfig_path),
            legacy_key: 'legacy_value',
        }
        lib_config = {'library_name': 'test_lib', 'options': [option_spec]}
        config_path = tmp_path / 'lib.yaml'
        _write_yaml(path=config_path, data=lib_config)

        env = create_mock_environment(antibiotic_names=['A'])
        with pytest.raises(RuntimeError, match='legacy top-level loader keys'):
            OptionLibraryLoader.load_library(
                library_config_path=str(config_path),
                env=env,
            )


class TestOptionLibraryLoaderConfigMerge:
    """Test config merging behavior."""

    def test_override_merges_with_defaults(self, tmp_path):
        """Test that overrides merge with defaults."""
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()

        default_config = {
            'antibiotic': 'A',
            'duration': 5,
            'extra_param': 'default_value'
        }
        default_config_path = tmp_path / 'block_option_default_config.yaml'
        _write_yaml(path=default_config_path, data=default_config)

        lib_config = {
            'library_name': 'test_lib',
            'options': [
                {
                    'option_name': 'OPT_1',
                    'option_type': 'block',
                    'option_subconfig_file': str(default_config_path),
                    'config_params_override': {'duration': 10},  # Only override duration
                }
            ]
        }
        lib_config_path = lib_dir / 'test_lib.yaml'
        _write_yaml(path=lib_config_path, data=lib_config)

        env = create_mock_environment(antibiotic_names=['A', 'B'], num_patients_per_time_step=1)
        _, resolved = OptionLibraryLoader.load_library(
            library_config_path=str(lib_config_path),
            env=env,
        )

        opt_resolved = resolved['options'][0]
        assert opt_resolved['merged_config']['antibiotic'] == 'A'  # From default
        assert opt_resolved['merged_config']['duration'] == 10  # From override
        assert opt_resolved['merged_config']['extra_param'] == 'default_value'  # From default
