"""evaluate.py, the released rhythm checkpoint and the web viewer - no GPU, no PhysioNet data."""
import asyncio
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, 'checkpoints', 'dualunet_rhythm_1m_e26')
sys.path.insert(0, ROOT)


def test_checkpoint_folder_is_complete_and_matches_its_hash():
    for name in ('best_model.keras', 'train_config.json', 'manifest.json', 'SHA256',
                 'TRAINING_LOG.md'):
        assert os.path.exists(os.path.join(CKPT_DIR, name)), name
    with open(os.path.join(CKPT_DIR, 'manifest.json')) as f:
        manifest = json.load(f)
    h = hashlib.sha256(open(os.path.join(CKPT_DIR, 'best_model.keras'), 'rb').read()).hexdigest()
    assert h == manifest['sha256'] == open(os.path.join(CKPT_DIR, 'SHA256')).read().split()[0]
    assert manifest['contract']['input'] == [2500, 3]
    for rel in manifest['decode_files']:
        assert os.path.exists(os.path.join(CKPT_DIR, rel)), rel


def test_evaluate_config_points_at_the_release():
    import evaluate
    assert os.path.exists(evaluate.resolve(evaluate.CHECKPOINT))
    assert os.path.exists(evaluate.resolve(evaluate.TRAIN_CONFIG))
    assert evaluate.DECODE is None                       # official = the config defaults
    assert 0 <= evaluate.PREDICT_OVERLAP_SECONDS < 10


def test_export_environment_reaches_the_rhythm_config():
    """In a fresh interpreter: the CONFIG block must be what ecgr.rhythm.config resolves."""
    code = ("import evaluate; evaluate.PREDICT_OVERLAP_SECONDS = 9.5; evaluate.export_environment();"
            "from ecgr.rhythm import config as rc; from ecgr.rhythm.predict import window_hop_samples;"
            "print(rc.PREDICT_OVERLAP_SECONDS, window_hop_samples(), rc.EC57_DIR)")
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3'))
    assert out.returncode == 0, out.stderr[-2000:]
    overlap, hop, ec57_dir = out.stdout.split()[-3:]
    assert float(overlap) == 9.5 and int(hop) == 130            # 0.5 s step -> 0.52 s grid
    assert ec57_dir == os.path.join(ROOT, 'eval_results', 'dualunet_rhythm_1m_e26', 'ec57')


@pytest.mark.parametrize('name', ['validation_tuned.json', 'overlap1_validation_tuned.json',
                                  'overlap1_af_context.json'])
def test_released_decode_files_decode(name):
    from ecgr.rhythm import config as rc, ec57, tune
    decode = tune.load_decode(os.path.join(CKPT_DIR, 'decode', name))
    hz, n = rc.SAMPLE_PROBS_HZ, 120
    rhythm = np.zeros((n * hz, rc.NUM_CLASSES), np.float32)
    rhythm[:, rc.SINUS] = 1.0
    rhythm[40 * hz:90 * hz] = 0.0
    rhythm[40 * hz:90 * hz, 1] = 1.0                    # 50 s of AFIB
    t = np.arange(0.4, n, 0.8)
    beats = dict(t=t, cls=np.ones(len(t), int), conf=np.ones(len(t), np.float32),
                 probs=np.tile([0.9, 0.05, 0.05], (len(t), 1)).astype(np.float32))
    episodes, symbols = ec57.decode_record(rhythm, np.zeros(n * hz), hz, beats, decode)
    af = [e for e in episodes if e['rhythm'] == 'AFIB']
    assert len(af) == 1 and 39 <= af[0]['start'] <= 41 and 89 <= af[0]['stop'] <= 91
    assert len(symbols) == len(t)


def test_afib_record_gate_drops_unsustained_af_in_long_records_only():
    from ecgr.rhythm import config as rc, ec57
    hz = rc.SAMPLE_PROBS_HZ
    decode = dict(beat_pp=False, afib_gate=True, min_seconds={'AFIB': 1}, merge_gap={}, smooth_seconds={})

    def record(seconds, af_spans, p_af=0.95):
        rhythm = np.zeros((seconds * hz, rc.NUM_CLASSES), np.float32)
        rhythm[:, rc.SINUS] = 1.0
        for a, b in af_spans:
            rhythm[a * hz:b * hz, rc.SINUS] = 1 - p_af
            rhythm[a * hz:b * hz, 1] = p_af
        return rhythm

    # 20 min record with only 10 s AF bursts -> gated away; a 40 s sustained one keeps them all
    r = record(1200, [(100, 110), (500, 512), (900, 908)])
    eps, _ = ec57.decode_record(r, np.zeros(len(r)), hz, None, decode)
    assert not [e for e in eps if e['rhythm'] == 'AFIB']
    r = record(1200, [(100, 110), (500, 540), (900, 908)])
    eps, _ = ec57.decode_record(r, np.zeros(len(r)), hz, None, decode)
    assert len([e for e in eps if e['rhythm'] == 'AFIB']) == 3
    # a sustained but weak episode does not confirm AF
    r = record(1200, [(500, 540)], p_af=0.5)
    eps, _ = ec57.decode_record(r, np.zeros(len(r)), hz, None, decode)
    assert not [e for e in eps if e['rhythm'] == 'AFIB']
    # AF-confirmed record + relaxed acceptance: a weak 8 s burst (p 0.58, under the 0.6
    # min_prob but still the argmax after the class prior) is accepted on the second pass only
    r = record(1200, [(500, 540)])
    r[900 * hz:908 * hz, rc.SINUS] = 0.42
    r[900 * hz:908 * hz, 1] = 0.58
    eps, _ = ec57.decode_record(r, np.zeros(len(r)), hz, None, decode)
    assert len([e for e in eps if e['rhythm'] == 'AFIB']) == 1
    relaxed = dict(decode, afib_gate={'relaxed': {'min_prob': {'AFIB': 0.5}}})
    eps, _ = ec57.decode_record(r, np.zeros(len(r)), hz, None, relaxed)
    assert len([e for e in eps if e['rhythm'] == 'AFIB']) == 2
    # a 60 s strip is below min_record: untouched
    r = record(60, [(10, 18)])
    eps, _ = ec57.decode_record(r, np.zeros(len(r)), hz, None, decode)
    assert len([e for e in eps if e['rhythm'] == 'AFIB']) == 1


def _viewer_app():
    import importlib.util
    spec = importlib.util.spec_from_file_location('viewer_app', os.path.join(ROOT, 'webservice',
                                                                             'app.py'))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def test_viewer_lists_a_run_and_serves_its_info(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    app = _viewer_app()
    app.check_classes()
    run = tmp_path / 'ec57' / 'smoke'
    (run / '_ann' / 'mitdb').mkdir(parents=True)
    np.savez(run / '_ann' / 'mitdb' / '100.npz', rhythm=np.eye(5)[[0] * 50 + [1] * 50],
             p_noise=np.zeros(100), fs=360, sig_len=3600, step_hz=10)
    (run / 'evaluate_config.json').write_text(json.dumps({'overlap_s': 5, 'decode': None}))

    async def go():
        client = TestClient(TestServer(app.make_app(str(tmp_path), str(tmp_path), 'none')))
        await client.start_server()
        try:
            runs = await (await client.get('/api/runs')).json()
            assert runs['runs'] == [str(run)]
            dbs = await (await client.get('/api/dbs', params={'run': str(run)})).json()
            recs = await (await client.get('/api/records', params={'run': str(run), 'db': 'mitdb'})).json()
            info = await (await client.get('/api/run_info', params={'run': str(run)})).json()
            page = await client.get('/')
            return dbs, recs, info, page.status, await page.text()
        finally:
            await client.close()

    dbs, recs, info, status, html = asyncio.run(go())
    assert dbs['dbs'] == ['mitdb'] and recs['records'] == ['100']
    assert info['config']['overlap_s'] == 5 and info['rhythm'] == [] and info['beats'] == []
    assert status == 200 and 'id="tr"' in html                 # the Run tab
    eps, length = app.pre_episodes(str(run / '_ann' / 'mitdb' / '100.npz'))
    assert length == 10.0 and [(e['rhythm'], e['start'], e['stop']) for e in eps] == [('AFIB', 5.0, 10.0)]
