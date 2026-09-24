"""Real pinned RCI → FSRT → JSPT provenance and offline replay gates."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.covariance_workflow import execute_covariance, replay_covariance
from ciw.investigation import create_investigation, inspect_investigation, replay_investigation
from ciw.session import Session, read_json


pytestmark = pytest.mark.skipif(not all(os.getenv('CIW_' + name + '_REPO') for name in ('RCI', 'FSRT', 'JSPT')),
                               reason='Requires three clean pinned scientific checkouts')
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def sources():
    return tuple(Path(os.environ['CIW_' + name + '_REPO']) for name in ('RCI', 'FSRT', 'JSPT'))


@pytest.fixture(scope='module')
def inputs():
    return json.loads((ROOT / 'examples/adapters/two-reservoir-covariance.json').read_text())


@pytest.fixture(scope='module')
def created(tmp_path_factory, sources, inputs):
    return create_investigation(deepcopy(inputs), *sources[:2], tmp_path_factory.mktemp('covariance-chain') / 'source', sys.executable)


@pytest.fixture(scope='module')
def mapped(tmp_path_factory, sources, created):
    params = json.loads((ROOT / 'examples/adapters/tank-covariance-map.json').read_text())
    return execute_covariance(created['workspace_file'], sources[2], params,
                              tmp_path_factory.mktemp('covariance-map') / 'mapped', sys.executable)


def test_calibration_provenance_and_native_parameterization_survive(created, inputs):
    assert created['status'] == 'completed'
    workspace = read_json(Path(created['workspace_file']))
    for source, sensor in zip(inputs['sensors'], workspace['run']['metadata']['rci_source']['sensors']):
        measurement = sensor['measurement']
        assert measurement['records'][0]['raw_record_b64'] == source['request']['inputs']['records'][0]['raw_record_b64']
        assert measurement['uncertainty']['covariance_basis'] == source['request']['inputs']['calibration']['covariance_basis']
        assert measurement['uncertainty']['parameterization']['name'] == 'scale_zero_raw'
        assert measurement['uncertainty']['covariance_coverage']['completeness_claimed'] is False
        assert measurement['records'][0]['acquisition_applicability']['applicable'] is True
    assert all(row['acquisition']['provenance'] == 'persisted.v2' for row in created['calibration'])
    assert all(row['serving']['expired'] for row in inspect_investigation(created['workspace_file'], '2030-01-01T00:00:00Z')['calibration'])


def test_fsrt_stages_and_full_covariance_are_retained(created):
    result = created['results'][-1]
    assert result['operation_id'] == 'fsrt.tank-reconstruct.v2'
    artifacts = result['data']['covariance_artifacts']
    assert set(artifacts) == {'observation', 'prior', 'declared_total', 'innovation', 'posterior', 'reconciled'}
    assert artifacts['reconciled']['matrix'] == result['data']['estimate']['covariance']
    assert artifacts['reconciled']['matrix'][0][1] != 0
    assert artifacts['posterior']['matrix'] == result['data']['unprojected_estimate']['covariance']
    assert len(artifacts['observation']['provenance']['metadata']['uncertainty_context']) == 2
    assert result['verification_id'] is None and result['verification_status'] == 'not_verified'


def test_jspt_coordinate_operation_uses_retained_joint_state(mapped, created):
    assert mapped['status'] == 'completed'
    result = mapped['results'][-1]
    assert mapped['results'][0] == created['results'][0]
    assert result['operation_id'] == 'jspt.covariance-propagate.v1'
    data = result['data']
    source = created['results'][0]['data']['covariance_artifacts']['reconciled']
    assert data['input_covariance'] == source
    jacobian = np.array([[1., 1.], [1., -1.]])
    matrix = np.array(source['matrix'])
    np.testing.assert_allclose(data['output_covariance']['matrix'], jacobian @ matrix @ jacobian.T, rtol=1e-12, atol=1e-15)
    assert not np.allclose(jacobian @ matrix @ jacobian.T, jacobian @ np.diag(np.diag(matrix)) @ jacobian.T, rtol=1e-12, atol=1e-15)
    assert data['output_covariance']['provenance']['source_covariance_ids'] == [source['covariance_id']]
    assert result['parameters']['source_result_id'] == created['results'][0]['result_id']
    assert result['evidence_id'] == created['evidence_id']
    assert result['verification_id'] is None and data['checks']['jacobian_verified'] is False


def test_saved_chain_inspects_without_importing_domain_engines(mapped, tmp_path):
    before = Path(mapped['workspace_file']).read_bytes()
    session = Session.from_workspace(Path(mapped['workspace_file']), tmp_path / 'opened')
    assert session.results[mapped['results'][-1]['result_id']] == mapped['results'][-1]
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in ('instrument_chain', 'set_lcm', 'sensitivity'))
    assert Path(mapped['workspace_file']).read_bytes() == before
    assert inspect_investigation(mapped['workspace_file'])['evidence_id'] == mapped['evidence_id']


def test_covariance_replay_preserves_old_result_and_allocates_new_identities(mapped, sources, tmp_path):
    old_bytes = Path(mapped['workspace_file']).read_bytes()
    replay = replay_covariance(mapped['workspace_file'], sources[2], tmp_path / 'replayed', sys.executable)
    assert replay['replay_data_digest_matches'] is True
    assert replay['results'][:-1] == mapped['results']
    assert replay['results'][-1]['execution_id'] != mapped['results'][-1]['execution_id']
    assert replay['results'][-1]['result_id'] != mapped['results'][-1]['result_id']
    assert replay['results'][-1]['data'] == mapped['results'][-1]['data']
    assert Path(mapped['workspace_file']).read_bytes() == old_bytes


def test_v2_calibration_replays_after_serving_expiry(created, sources, tmp_path):
    summary = inspect_investigation(created['workspace_file'], '2030-01-01T00:00:00Z')
    assert all(item['expired'] and item['applicable_at_acquisition'] for item in summary['calibration'])
    replay = replay_investigation(created['workspace_file'], *sources[:2], tmp_path / 'replayed-estimator', sys.executable)
    assert replay['replay_data_digest_matches'] is True


def test_common_reference_cannot_silently_enter_independent_assembly_composition(inputs, sources, tmp_path):
    altered = deepcopy(inputs)
    left = altered['sensors'][0]['request']['inputs']['calibration']['covariance_basis']
    right = altered['sensors'][1]['request']['inputs']['calibration']['covariance_basis']
    reference = next(block for block in left['parameter_components'] if block['kind'] == 'reference_standard')
    target = next(block for block in right['parameter_components'] if block['kind'] == 'reference_standard')
    for key in ('dependency_ids', 'shared_source_ids', 'evidence_ids'):
        target[key] = deepcopy(reference[key])
    destination = tmp_path / 'must-not-write'
    with pytest.raises(AdapterRefusal, match='share uncertainty sources'):
        create_investigation(altered, *sources[:2], destination, sys.executable)
    assert not destination.exists()


@pytest.mark.parametrize('flag', ['prior_independent_of_observations', 'declared_total_independent_of_observations', 'prior_independent_of_declared_total'])
def test_model_dependence_is_not_assumed(inputs, sources, tmp_path, flag):
    altered = deepcopy(inputs)
    altered['model_independence'][flag] = False
    with pytest.raises(AdapterRefusal, match='mutually independent'):
        create_investigation(altered, *sources[:2], tmp_path / flag, sys.executable)
    assert not (tmp_path / flag).exists()


def test_rehashed_source_dependency_mismatch_fails_before_destination_writes(mapped, tmp_path):
    from ciw.operations.runner import seal
    workspace = read_json(Path(mapped['workspace_file']))
    result = workspace['results'][-1]
    result['parameters']['source_result_id'] = result['result_id']
    seal(result)
    for execution in workspace['executions']:
        if execution['execution_id'] == result['execution_id']:
            execution['parameters'] = deepcopy(result['parameters'])
            seal(execution)
    path = tmp_path / 'cycle.json'
    path.write_text(json.dumps(workspace))
    output = tmp_path / 'must-not-write'
    with pytest.raises(ValueError):
        Session.from_workspace(path, output)
    assert not output.exists()


def test_historical_pins_remain_replayable_after_runtime_upgrade(monkeypatch, sources, tmp_path):
    legacy = [os.getenv('CIW_' + name + '_LEGACY_REPO') for name in ('RCI', 'FSRT')]
    if not all(legacy):
        pytest.skip('Historical clean pins are provided by scripts/check_adapters.py')
    import ciw.investigation as module
    from ciw.adapters.subprocess import PinnedSubprocessAdapter
    pins = json.loads((ROOT / 'src/ciw/adapter-runtimes.json').read_text())
    original_runtime = module._runtime

    def historic(name, repo, python_executable=None, expected=None):
        if expected is not None:
            return original_runtime(name, repo, python_executable, expected)
        spec = pins[name]['historical'][0]
        return PinnedSubprocessAdapter(repo, spec['revision'], spec['module'], python_executable=python_executable or sys.executable)

    with monkeypatch.context() as patch:
        patch.setattr(module, '_runtime', historic)
        v1 = json.loads((ROOT / 'examples/adapters/two-reservoir.json').read_text())
        old = create_investigation(v1, *legacy, tmp_path / 'old', sys.executable)
    replay = replay_investigation(old['workspace_file'], *legacy, tmp_path / 'old-replayed', sys.executable)
    assert replay['replay_data_digest_matches'] is True
    assert replay['results'][0] == old['results'][0]
    assert replay['results'][-1]['runtime']['revision'] == pins['fsrt']['historical'][0]['revision']
    assert replay['results'][-1]['runtime']['revision'] != pins['fsrt']['revision']
