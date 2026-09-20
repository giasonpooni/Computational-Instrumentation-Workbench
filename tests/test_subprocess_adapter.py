"""Real child-process checks for the generic scientific adapter boundary."""

from __future__ import annotations

import importlib._bootstrap_external
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.adapters.subprocess import PinnedSubprocessAdapter


_ENDPOINT = '''import importlib.util, json, pathlib, sys, time
request = json.load(sys.stdin)
assert request['schema'] == 'ciw.adapter-request.v1'
operation = request['operation_id']
if operation == 'timeout':
    time.sleep(10)
elif operation == 'stdout_flood':
    sys.stdout.write('x' * 100000)
elif operation == 'stderr_flood':
    sys.stderr.write('x' * 100000)
elif operation == 'small_flood_then_wait':
    sys.stdout.write('x' * 1500)
    sys.stdout.flush()
    time.sleep(10)
elif operation == 'exit':
    sys.exit(3)
elif operation == 'literal':
    sys.stdout.write(request['inputs']['literal'])
elif operation == 'mutate':
    pathlib.Path(__file__).write_text('# changed during operation')
    print(json.dumps({'schema':'ciw.adapter-response.v1','status':'ok','data':{}}))
elif operation == 'refuse':
    print(json.dumps({'schema':'ciw.adapter-response.v1','status':'refused',
                     'refusal':{'code':'MODEL_DISAGREEMENT','message':'Declared model disagrees'}}))
else:
    print(json.dumps({'schema':'ciw.adapter-response.v1','status':'ok',
                     'data':{'inputs':request['inputs'], 'isolated':sys.flags.isolated,
                             'poison_absent':importlib.util.find_spec('ciw_test_poison') is None}}))
'''


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / 'runtime'
    (root / 'src' / 'example').mkdir(parents=True)
    (root / 'src' / 'example' / '__init__.py').write_text('')
    (root / 'src' / 'example' / 'endpoint.py').write_text(_ENDPOINT)
    (root / '.gitignore').write_text('.venv/\n__pycache__/\nsrc/ignored.py\n')
    # A checkout-root module must not shadow the actual standard library.
    (root / 'json.py').write_text('raise RuntimeError("cwd shadow executed")\n')
    git(root, 'init', '-q')
    # Pin the fixture's exact working bytes on Windows as well as POSIX;
    # user/global autocrlf settings must not normalize the committed blobs.
    git(root, 'config', 'core.autocrlf', 'false')
    git(root, 'add', '.')
    git(root, '-c', 'user.name=CIW Test', '-c', 'user.email=ciw@example.test', 'commit', '-qm', 'fixture')
    return root, git(root, 'rev-parse', 'HEAD')


def bound(checkout: tuple[Path, str], **kwargs: object) -> PinnedSubprocessAdapter:
    root, revision = checkout
    return PinnedSubprocessAdapter(root, revision, 'example.endpoint', **kwargs)


def test_roundtrip_and_runtime_identity(checkout: tuple[Path, str]) -> None:
    adapter = bound(checkout)
    before = adapter.runtime_identity()
    assert adapter.invoke('evaluate.v1', {'state': [1, 2.5]}) == {
        'inputs': {'state': [1, 2.5]}, 'isolated': 1, 'poison_absent': True,
    }
    assert before == adapter.runtime_identity()
    assert before['revision'] == checkout[1]
    assert len(before['python_sha256']) == 64
    assert before['dependencies'].keys() == {'numpy', 'scipy'}
    # Rebinding requires an explicit local operator action and expected pins.
    restored = bound(
        checkout, expected_python_sha256=before['python_sha256'],
        expected_python_version=before['python_version'], expected_dependencies=before['dependencies'],
    )
    assert restored.runtime_identity() == before


def test_cwd_and_pythonpath_cannot_supply_modules(
    checkout: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    poison = tmp_path / 'poison'
    poison.mkdir()
    (poison / 'ciw_test_poison.py').write_text('raise RuntimeError("unbound module")')
    monkeypatch.chdir(poison)
    monkeypatch.setenv('PYTHONPATH', str(poison))
    assert bound(checkout).invoke('evaluate.v1', {})['poison_absent'] is True


def test_domain_refusal_has_no_result(checkout: tuple[Path, str]) -> None:
    with pytest.raises(AdapterRefusal) as refusal:
        bound(checkout).invoke('refuse', {})
    assert refusal.value.code == 'MODEL_DISAGREEMENT'


@pytest.mark.parametrize('reason_code', ['calibration_missing', 'calibration_expired', 'calibration_mismatch'])
def test_domain_refusal_retains_optional_reason(checkout: tuple[Path, str], reason_code: str) -> None:
    detail = {'code': 'calibration_unavailable', 'message': 'Calibration unavailable', 'reason_code': reason_code}
    payload = {'schema': 'ciw.adapter-response.v1', 'status': 'refused', 'refusal': detail}
    with pytest.raises(AdapterRefusal) as refusal:
        bound(checkout).invoke('literal', {'literal': json.dumps(payload)})
    assert refusal.value.code == 'calibration_unavailable'
    assert refusal.value.reason_code == reason_code
    assert refusal.value.to_dict() == detail


def test_refusal_reason_survives_session_save_and_reopen(tmp_path: Path) -> None:
    from ciw.instruments import make_demo_run
    from ciw.operations.registry import Operation, OperationRegistry
    from ciw.session import Session

    def refuse(run: dict, parameters: dict) -> dict:
        raise AdapterRefusal('calibration_unavailable', 'Calibration unavailable', reason_code='calibration_missing')

    operations = OperationRegistry()
    operations.register(Operation('statistics.v1', 'analysis', refuse, lambda: {'version': 'test'}))
    session = Session(make_demo_run(), tmp_path / 'source', operations=operations)
    reply = session.handle({
        'protocol_version': 1, 'request_id': 'reason-code', 'type': 'operation.execute',
        'payload': {'operation_id': 'statistics.v1', 'parameters': {}},
    })
    assert reply['payload']['execution']['refusal']['reason_code'] == 'calibration_missing'
    assert reply['payload']['result'] is None
    saved = session.save_workspace(tmp_path / 'workspace.json')
    reopened = Session.from_workspace(saved, tmp_path / 'reopened')
    assert reopened.executions == session.executions
    assert reopened.results == {}


@pytest.mark.parametrize('payload', [
    'not json',
    '{"schema":"ciw.adapter-response.v1","status":"ok","data":{"x":NaN}}',
    '{"schema":"ciw.adapter-response.v1","status":"ok","data":{"x":1e400}}',
    '{"schema":"ciw.adapter-response.v1","status":"ok","data":{},"data":{"x":1}}',
    '{"schema":"ciw.adapter-response.v1","status":"ok","data":[]}',
    '{"schema":"ciw.adapter-response.v1","status":"ok","data":{},"extra":true}',
    '{"schema":"ciw.adapter-response.v1","status":"refused","data":{},"refusal":{"code":"X","message":"x"}}',
    '{"schema":"ciw.adapter-response.v1","status":"refused","refusal":{"code":"","message":"x"}}',
    '{"schema":"ciw.adapter-response.v1","status":"refused","refusal":{"code":"X","message":"x","reason_code":""}}',
    '{"schema":"ciw.adapter-response.v1","status":"refused","refusal":{"code":"X","message":"x","reason_code":" "}}',
    '{"schema":"ciw.adapter-response.v1","status":"refused","refusal":{"code":"X","message":"x","reason_code":null}}',
    '{"schema":"ciw.adapter-response.v1","status":"refused","refusal":{"code":"X","message":"x","unexpected":"x"}}',
    '[]',
])
def test_malformed_and_nonfinite_output_is_refused(checkout: tuple[Path, str], payload: str) -> None:
    with pytest.raises(AdapterRefusal) as refusal:
        bound(checkout).invoke('literal', {'literal': payload})
    assert refusal.value.code == 'MALFORMED_RESPONSE'


@pytest.mark.parametrize('operation', ['stdout_flood', 'stderr_flood', 'small_flood_then_wait'])
def test_both_output_pipes_are_bounded(checkout: tuple[Path, str], operation: str) -> None:
    with pytest.raises(AdapterRefusal) as refusal:
        bound(checkout, max_output_bytes=1024).invoke(operation, {})
    assert refusal.value.code == 'OUTPUT_LIMIT'


def test_timeout_and_nonzero_exit_are_refused(checkout: tuple[Path, str]) -> None:
    adapter = bound(checkout)
    with pytest.raises(AdapterRefusal) as failed:
        adapter.invoke('exit', {})
    assert failed.value.code == 'RUNTIME_FAILED'
    adapter.timeout_seconds = 0.3
    with pytest.raises(AdapterRefusal) as timed_out:
        adapter.invoke('timeout', {})
    assert timed_out.value.code == 'TIMEOUT'


@pytest.mark.parametrize('kind', ['tracked', 'untracked', 'ignored', 'staged', 'head'])
def test_source_drift_refuses_execution(checkout: tuple[Path, str], kind: str) -> None:
    adapter = bound(checkout)
    root, _ = checkout
    if kind == 'tracked':
        (root / 'src' / 'example' / 'endpoint.py').write_text('raise SystemExit(0)')
    elif kind == 'untracked':
        (root / 'src' / 'example' / 'extra.py').write_text('')
    elif kind == 'ignored':
        (root / 'src' / 'ignored.py').write_text('')
    elif kind == 'staged':
        git(root, 'update-index', '--chmod=+x', 'json.py')
    else:
        git(root, '-c', 'user.name=CIW Test', '-c', 'user.email=ciw@example.test',
            'commit', '--allow-empty', '-qm', 'new head')
    with pytest.raises(AdapterRefusal) as refusal:
        adapter.invoke('evaluate.v1', {})
    assert refusal.value.code == 'SOURCE_PIN_MISMATCH'


def test_pin_check_reads_bytes_even_when_git_assumes_unchanged(checkout: tuple[Path, str]) -> None:
    adapter = bound(checkout)
    root, _ = checkout
    git(root, 'update-index', '--assume-unchanged', 'src/example/endpoint.py')
    (root / 'src' / 'example' / 'endpoint.py').write_text('raise SystemExit(0)')
    with pytest.raises(AdapterRefusal) as refusal:
        adapter.runtime_identity()
    assert refusal.value.code == 'SOURCE_PIN_MISMATCH'


def test_cache_and_virtualenv_files_are_excluded(checkout: tuple[Path, str]) -> None:
    root, _ = checkout
    for folder in ('.venv', 'src/example/__pycache__', '.pytest_cache'):
        path = root / folder
        path.mkdir(parents=True)
        (path / 'temporary').write_text('cache')
    assert bound(checkout).invoke('evaluate.v1', {})['isolated'] == 1


def test_forged_timestamp_matching_bytecode_cannot_replace_pinned_source(
    checkout: tuple[Path, str],
) -> None:
    root, _ = checkout
    source = root / 'src' / 'example' / 'endpoint.py'
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir()
    forged = compile(
        "print('{\"schema\":\"ciw.adapter-response.v1\",\"status\":\"ok\",\"data\":{\"forged\":true}}')",
        str(source), 'exec',
    )
    metadata = source.stat()
    cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
        forged, int(metadata.st_mtime), metadata.st_size,
    ))
    assert bound(checkout).invoke('evaluate.v1', {'original': True})['inputs'] == {'original': True}
    assert cache.exists()  # User caches are never removed or changed.


def test_mutation_during_execution_cannot_produce_a_result(checkout: tuple[Path, str]) -> None:
    with pytest.raises(AdapterRefusal) as refusal:
        bound(checkout).invoke('mutate', {})
    assert refusal.value.code == 'SOURCE_PIN_MISMATCH'


@pytest.mark.parametrize('pin', [
    {'expected_python_sha256': '0' * 64},
    {'expected_python_version': '0.0.0'},
    {'expected_dependencies': {'numpy': '0.0.0', 'scipy': None}},
])
def test_expected_runtime_pins_are_enforced(checkout: tuple[Path, str], pin: dict[str, object]) -> None:
    with pytest.raises(AdapterRefusal) as refusal:
        bound(checkout, **pin)
    assert refusal.value.code == 'RUNTIME_PIN_MISMATCH'


def test_executable_drift_and_invalid_input_refuse(
    checkout: tuple[Path, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = bound(checkout)
    with pytest.raises(AdapterRefusal) as invalid:
        adapter.invoke('evaluate.v1', {'bad': float('nan')})
    assert invalid.value.code == 'INVALID_INPUT'
    with pytest.raises(AdapterRefusal) as invalid_key:
        adapter.invoke('evaluate.v1', {'nested': {1: 'value'}})
    assert invalid_key.value.code == 'INVALID_INPUT'
    monkeypatch.setattr(adapter, '_executable_digest', lambda: 'f' * 64)
    with pytest.raises(AdapterRefusal) as mismatch:
        adapter.invoke('evaluate.v1', {})
    assert mismatch.value.code == 'RUNTIME_PIN_MISMATCH'


def test_arguments_do_not_allow_revision_or_command_injection(checkout: tuple[Path, str]) -> None:
    root, revision = checkout
    with pytest.raises(ValueError):
        PinnedSubprocessAdapter(root, 'HEAD', 'example.endpoint')
    with pytest.raises(ValueError):
        PinnedSubprocessAdapter(root, revision, 'example.endpoint; touch /tmp/ciw')
    with pytest.raises(ValueError):
        PinnedSubprocessAdapter(root, revision, 'example.endpoint', source_root='../')
    with pytest.raises(AdapterRefusal):
        PinnedSubprocessAdapter(root, revision, 'example.missing')
