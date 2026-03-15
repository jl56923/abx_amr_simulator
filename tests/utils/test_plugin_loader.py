from pathlib import Path

import pytest

from abx_amr_simulator.core import PatientGeneratorBase
from abx_amr_simulator.utils.plugin_loader import load_plugin_component


def _write_loader_file(*, tmp_path: Path, filename: str, content: str) -> str:
    loader_path = tmp_path / filename
    loader_path.write_text(data=content, encoding='utf-8')
    return str(loader_path)


def test_returns_none_when_no_plugin_key() -> None:
    component = load_plugin_component(
        component_config={},
        expected_base_class=PatientGeneratorBase,
        default_loader_fn_name='load_patient_generator_component',
        config_dir_hint=None,
    )

    assert component is None


def test_loads_from_filesystem_absolute_path(tmp_path: Path) -> None:
    loader_file = _write_loader_file(
        tmp_path=tmp_path,
        filename='stub_pg_loader.py',
        content='''
from abx_amr_simulator.core import PatientGeneratorBase
import numpy as np


class StubPatientGenerator(PatientGeneratorBase):
    PROVIDES_ATTRIBUTES = []
    visible_patient_attributes = []

    def __init__(self, config):
        self.config = config

    def sample(self, n, true_amr_levels, rng):
        return []

    def observe(self, patients):
        return np.array([], dtype=float)

    def obs_dim(self, num_patients):
        return 0


def load_patient_generator_component(config):
    return StubPatientGenerator(config=config)
''',
    )
    component = load_plugin_component(
        component_config={'plugin': {'loader_module': loader_file}},
        expected_base_class=PatientGeneratorBase,
        default_loader_fn_name='load_patient_generator_component',
        config_dir_hint=None,
    )

    assert isinstance(component, PatientGeneratorBase)


def test_loads_from_filesystem_relative_path_with_hint(tmp_path: Path) -> None:
    _write_loader_file(
        tmp_path=tmp_path,
        filename='stub_pg_loader_rel.py',
        content='''
from abx_amr_simulator.core import PatientGeneratorBase
import numpy as np


class StubPatientGenerator(PatientGeneratorBase):
    PROVIDES_ATTRIBUTES = []
    visible_patient_attributes = []

    def sample(self, n, true_amr_levels, rng):
        return []

    def observe(self, patients):
        return np.array([], dtype=float)

    def obs_dim(self, num_patients):
        return 0


def load_patient_generator_component(config):
    return StubPatientGenerator()
''',
    )

    component = load_plugin_component(
        component_config={'plugin': {'loader_module': 'stub_pg_loader_rel.py'}},
        expected_base_class=PatientGeneratorBase,
        default_loader_fn_name='load_patient_generator_component',
        config_dir_hint=str(tmp_path),
    )

    assert isinstance(component, PatientGeneratorBase)


def test_uses_default_loader_fn_name(tmp_path: Path) -> None:
    loader_file = _write_loader_file(
        tmp_path=tmp_path,
        filename='stub_pg_loader_default.py',
        content='''
from abx_amr_simulator.core import PatientGeneratorBase
import numpy as np


class StubPatientGenerator(PatientGeneratorBase):
    PROVIDES_ATTRIBUTES = []
    visible_patient_attributes = []

    def sample(self, n, true_amr_levels, rng):
        return []

    def observe(self, patients):
        return np.array([], dtype=float)

    def obs_dim(self, num_patients):
        return 0


def load_patient_generator_component(config):
    return StubPatientGenerator()
''',
    )

    component = load_plugin_component(
        component_config={'plugin': {'loader_module': loader_file}},
        expected_base_class=PatientGeneratorBase,
        default_loader_fn_name='load_patient_generator_component',
        config_dir_hint=None,
    )

    assert isinstance(component, PatientGeneratorBase)


def test_uses_custom_loader_fn_name(tmp_path: Path) -> None:
    loader_file = _write_loader_file(
        tmp_path=tmp_path,
        filename='stub_pg_loader_custom.py',
        content='''
from abx_amr_simulator.core import PatientGeneratorBase
import numpy as np


class StubPatientGenerator(PatientGeneratorBase):
    PROVIDES_ATTRIBUTES = []
    visible_patient_attributes = []

    def sample(self, n, true_amr_levels, rng):
        return []

    def observe(self, patients):
        return np.array([], dtype=float)

    def obs_dim(self, num_patients):
        return 0


def build_custom_pg(config):
    return StubPatientGenerator()
''',
    )

    component = load_plugin_component(
        component_config={
            'plugin': {
                'loader_module': loader_file,
                'loader_function': 'build_custom_pg',
            }
        },
        expected_base_class=PatientGeneratorBase,
        default_loader_fn_name='load_patient_generator_component',
        config_dir_hint=None,
    )

    assert isinstance(component, PatientGeneratorBase)


def test_raises_if_loader_module_key_missing() -> None:
    with pytest.raises(
        expected_exception=ValueError,
        match='missing required key',
    ):
        load_plugin_component(
            component_config={'plugin': {}},
            expected_base_class=PatientGeneratorBase,
            default_loader_fn_name='load_patient_generator_component',
            config_dir_hint=None,
        )


def test_raises_if_module_not_importable(tmp_path: Path) -> None:
    bad_loader_file = _write_loader_file(
        tmp_path=tmp_path,
        filename='bad_loader.py',
        content='def this_is_bad_python(\n',
    )

    with pytest.raises(expected_exception=ImportError, match='Failed to import plugin loader module'):
        load_plugin_component(
            component_config={'plugin': {'loader_module': bad_loader_file}},
            expected_base_class=PatientGeneratorBase,
            default_loader_fn_name='load_patient_generator_component',
            config_dir_hint=None,
        )


def test_raises_if_function_not_found(tmp_path: Path) -> None:
    loader_file = _write_loader_file(
        tmp_path=tmp_path,
        filename='stub_pg_loader_missing_fn.py',
        content='''
from abx_amr_simulator.core import PatientGeneratorBase
import numpy as np


class StubPatientGenerator(PatientGeneratorBase):
    PROVIDES_ATTRIBUTES = []
    visible_patient_attributes = []

    def sample(self, n, true_amr_levels, rng):
        return []

    def observe(self, patients):
        return np.array([], dtype=float)

    def obs_dim(self, num_patients):
        return 0
''',
    )

    with pytest.raises(expected_exception=AttributeError, match='Plugin loader function not found'):
        load_plugin_component(
            component_config={'plugin': {'loader_module': loader_file}},
            expected_base_class=PatientGeneratorBase,
            default_loader_fn_name='load_patient_generator_component',
            config_dir_hint=None,
        )


def test_raises_if_return_type_wrong(tmp_path: Path) -> None:
    loader_file = _write_loader_file(
        tmp_path=tmp_path,
        filename='stub_pg_loader_wrong_type.py',
        content='''
def load_patient_generator_component(config):
    return 7
''',
    )

    with pytest.raises(expected_exception=TypeError, match="Expected instance of 'PatientGeneratorBase', got 'int'"):
        load_plugin_component(
            component_config={'plugin': {'loader_module': loader_file}},
            expected_base_class=PatientGeneratorBase,
            default_loader_fn_name='load_patient_generator_component',
            config_dir_hint=None,
        )


def test_raises_if_relative_path_without_hint() -> None:
    with pytest.raises(
        expected_exception=ValueError,
        match='cannot be resolved as a relative filesystem path because no',
    ):
        load_plugin_component(
            component_config={'plugin': {'loader_module': 'stub_pg_loader_rel_missing_hint.py'}},
            expected_base_class=PatientGeneratorBase,
            default_loader_fn_name='load_patient_generator_component',
            config_dir_hint=None,
        )
