"""Unit tests for OptionLibraryLoader class."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock

from abx_amr_simulator.hrl import OptionBase, OptionLibrary, OptionLibraryLoader
import numpy as np


class SimpleBlockOption(OptionBase):
    """Simple option for testing."""
    REQUIRES_OBSERVATION_ATTRIBUTES = []
    REQUIRES_AMR_LEVELS = False

    def __init__(self, name: str, antibiotic: str, duration: int):
        super().__init__(name=name, k=duration)
        self.antibiotic = antibiotic

    def decide(self, env_state, antibiotic_names):
        num_patients = env_state['num_patients']
        try:
            idx = antibiotic_names.index(self.antibiotic)
        except ValueError:
            raise ValueError(f"Antibiotic '{self.antibiotic}' not found")
        return np.full(num_patients, idx, dtype=np.int32)


def create_loader_module(tmpdir, option_type='block'):
    """Helper to create a loader module file."""
    from abx_amr_simulator.hrl import OptionBase
    import numpy as np
    
    loader_code = f"""
import numpy as np
from abx_amr_simulator.hrl import OptionBase

class BlockOption(OptionBase):
    REQUIRES_OBSERVATION_ATTRIBUTES = []
    REQUIRES_AMR_LEVELS = False
    
    def __init__(self, name, antibiotic, duration):
        super().__init__(name=name, k=duration)
        self.antibiotic = antibiotic
    
    def decide(self, env_state, antibiotic_names):
        num_patients = env_state['num_patients']
        try:
            idx = antibiotic_names.index(self.antibiotic)
        except ValueError:
            raise ValueError(f"Antibiotic '{{self.antibiotic}}' not found")
        return np.full(num_patients, idx, dtype=np.int32)

def load_{option_type}_option(name, config):
    antibiotic = config.get('antibiotic', 'A')
    duration = config.get('duration', 5)
    return BlockOption(name=name, antibiotic=antibiotic, duration=duration)
"""
    
    module_path = tmpdir / f'{option_type}_option_loader.py'
    module_path.write_text(loader_code, encoding='utf-8')
    return module_path


class TestOptionLibraryLoaderBasic:
    """Basic tests for OptionLibraryLoader."""

    def test_load_library_from_yaml(self, tmp_path):
        """Test loading library from YAML config."""
        # Create directory structure
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()
        
        option_types_dir = tmp_path / 'option_types' / 'block'
        option_types_dir.mkdir(parents=True)
        
        # Create loader module
        create_loader_module(option_types_dir, 'block')
        
        # Create default config for block option
        default_config = {'antibiotic': 'A', 'duration': 5}
        default_config_path = option_types_dir / 'block_option_default_config.yaml'
        default_config_path.write_text(yaml.dump(default_config), encoding='utf-8')
        
        # Create library meta-config with ABSOLUTE paths (not relative)
        lib_config = {
            'library_name': 'test_lib',
            'description': 'Test library',
            'options': [
                {
                    'option_name': 'A_5',
                    'option_type': 'block',
                    'option_subconfig_file': str(default_config_path),
                    'loader_module': str(option_types_dir / 'block_option_loader.py'),
                    'config_params_override': {'antibiotic': 'A', 'duration': 5},
                }
            ]
        }
        lib_config_path = lib_dir / 'test_lib.yaml'
        lib_config_path.write_text(yaml.dump(lib_config), encoding='utf-8')
        
        # Load library
        lib, resolved = OptionLibraryLoader.load_library(str(lib_config_path))
        
        # Verify
        assert lib.name == 'test_lib'
        assert len(lib) == 1
        assert 'A_5' in lib.options
        assert resolved['library_name'] == 'test_lib'
        assert len(resolved['options']) == 1

    def test_load_library_missing_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            OptionLibraryLoader.load_library('/nonexistent/path/config.yaml')

    def test_load_library_empty_config(self, tmp_path):
        """Test loading from empty config."""
        lib_config_path = tmp_path / 'empty.yaml'
        lib_config_path.write_text('', encoding='utf-8')
        
        with pytest.raises(ValueError):
            OptionLibraryLoader.load_library(str(lib_config_path))

    def test_load_library_no_options(self, tmp_path):
        """Test loading config with no options."""
        lib_config = {
            'library_name': 'empty_lib',
            'options': []
        }
        lib_config_path = tmp_path / 'empty_lib.yaml'
        lib_config_path.write_text(yaml.dump(lib_config), encoding='utf-8')
        
        with pytest.raises(ValueError):
            OptionLibraryLoader.load_library(str(lib_config_path))


class TestOptionLibraryLoaderMultipleOptions:
    """Test loading multiple options."""

    def test_load_multiple_options(self, tmp_path):
        """Test loading library with multiple options."""
        # Setup directories
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()
        
        option_types_dir = tmp_path / 'option_types' / 'block'
        option_types_dir.mkdir(parents=True)
        
        # Create loader module
        create_loader_module(option_types_dir, 'block')
        
        # Create default config
        default_config = {'antibiotic': 'A', 'duration': 5}
        default_config_path = option_types_dir / 'block_option_default_config.yaml'
        default_config_path.write_text(yaml.dump(default_config), encoding='utf-8')
        
        loader_path = option_types_dir / 'block_option_loader.py'
        
        # Create library meta-config with 3 options using ABSOLUTE paths
        lib_config = {
            'library_name': 'test_lib',
            'options': [
                {
                    'option_name': f'OPT_{i}',
                    'option_type': 'block',
                    'option_subconfig_file': str(default_config_path),
                    'loader_module': str(loader_path),
                    'config_params_override': {'antibiotic': 'A', 'duration': i*5},
                }
                for i in range(1, 4)
            ]
        }
        lib_config_path = lib_dir / 'test_lib.yaml'
        lib_config_path.write_text(yaml.dump(lib_config), encoding='utf-8')
        
        # Load library
        lib, resolved = OptionLibraryLoader.load_library(str(lib_config_path))
        
        # Verify
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
        lib_config_path.write_text(yaml.dump(lib_config), encoding='utf-8')
        
        with pytest.raises(RuntimeError) as exc_info:
            OptionLibraryLoader.load_library(str(lib_config_path))
        
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
                    'loader_module': 'nonexistent/loader.py',
                }
            ]
        }
        lib_config_path = lib_dir / 'test_lib.yaml'
        lib_config_path.write_text(yaml.dump(lib_config), encoding='utf-8')
        
        with pytest.raises(RuntimeError):
            OptionLibraryLoader.load_library(str(lib_config_path))

    def test_loader_missing_function(self, tmp_path):
        """Test error when loader module missing function."""
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()
        
        option_types_dir = tmp_path / 'option_types' / 'block'
        option_types_dir.mkdir(parents=True)
        
        # Create loader module without the required function
        loader_code = "def other_function(): pass"
        loader_path = option_types_dir / 'block_option_loader.py'
        loader_path.write_text(loader_code, encoding='utf-8')
        
        # Create default config
        default_config_path = option_types_dir / 'block_option_default_config.yaml'
        default_config_path.write_text(yaml.dump({}), encoding='utf-8')
        
        # Create library config using ABSOLUTE paths
        lib_config = {
            'library_name': 'test_lib',
            'options': [
                {
                    'option_name': 'OPT_1',
                    'option_type': 'block',
                    'option_subconfig_file': str(default_config_path),
                    'loader_module': str(loader_path),
                }
            ]
        }
        lib_config_path = lib_dir / 'test_lib.yaml'
        lib_config_path.write_text(yaml.dump(lib_config), encoding='utf-8')
        
        with pytest.raises(RuntimeError) as exc_info:
            OptionLibraryLoader.load_library(str(lib_config_path))
        
        assert 'load_block_option' in str(exc_info.value)


class TestOptionLibraryLoaderConfigMerge:
    """Test config merging behavior."""

    def test_override_merges_with_defaults(self, tmp_path):
        """Test that overrides merge with defaults."""
        lib_dir = tmp_path / 'option_libraries'
        lib_dir.mkdir()
        
        option_types_dir = tmp_path / 'option_types' / 'block'
        option_types_dir.mkdir(parents=True)
        
        create_loader_module(option_types_dir, 'block')
        
        # Create default config with multiple keys
        default_config = {
            'antibiotic': 'A',
            'duration': 5,
            'extra_param': 'default_value'
        }
        default_config_path = option_types_dir / 'block_option_default_config.yaml'
        default_config_path.write_text(yaml.dump(default_config), encoding='utf-8')
        
        # Create library config with partial override using ABSOLUTE paths
        lib_config = {
            'library_name': 'test_lib',
            'options': [
                {
                    'option_name': 'OPT_1',
                    'option_type': 'block',
                    'option_subconfig_file': str(default_config_path),
                    'loader_module': str(option_types_dir / 'block_option_loader.py'),
                    'config_params_override': {'duration': 10},  # Only override duration
                }
            ]
        }
        lib_config_path = lib_dir / 'test_lib.yaml'
        lib_config_path.write_text(yaml.dump(lib_config), encoding='utf-8')
        
        # Load and verify
        lib, resolved = OptionLibraryLoader.load_library(str(lib_config_path))
        
        # Check resolved config shows the merge
        opt_resolved = resolved['options'][0]
        assert opt_resolved['merged_config']['antibiotic'] == 'A'  # From default
        assert opt_resolved['merged_config']['duration'] == 10  # From override
        assert opt_resolved['merged_config']['extra_param'] == 'default_value'  # From default
