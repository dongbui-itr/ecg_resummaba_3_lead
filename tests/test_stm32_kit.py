"""STM32F746G-Discovery kit: MCU export, the PC <-> kit protocol, the virtual kit (= the
firmware's C code built for the PC) against the Python pipeline, the web tab's API.

Tests that need the exported model / the virtual kit binary / PhysioNet data skip without them
(python3 firmware/stm32f746g_disco/export_model.py, make -C firmware/stm32f746g_disco/host).
"""
import asyncio
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'webservice')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import stm32_link as sl  # noqa: E402

MODEL_DIR = os.path.join(ROOT, 'firmware', 'stm32f746g_disco', 'Model')
MITDB = os.path.join(os.environ.get('ECGR_PHYSIONET_DIR', '/media/Project/ECG/PhysionetData'), 'mitdb')


def test_crc_and_frames_survive_garbage_and_a_corrupt_frame():
    assert sl.crc16(b'123456789') == 0x29B1 == sl._crc16_np(b'123456789')   # CRC-16/CCITT-FALSE
    good = [sl.encode(sl.RESULT, 7, b'abc'), sl.encode(sl.LOG, 8, b'')]
    bad = bytearray(sl.encode(sl.INFO, 9, b'xyz'))
    bad[7] ^= 0xFF
    r = sl.FrameReader()
    frames = r.feed(b'\x00\xa5noise' + good[0][:5]) + r.feed(good[0][5:] + bytes(bad) + good[1])
    assert frames == [(sl.RESULT, 7, b'abc'), (sl.LOG, 8, b'')]
    assert r.crc_errors == 1


def test_chunk_layout():
    uv = np.arange(2250 * 3, dtype=np.int16).reshape(2250, 3)
    p = sl.pack_chunk(5, uv, [1] * 9, [(10, 1), (300, 3)])
    assert len(p) == 10 + 2250 * 3 * 2 + 9 + 2 * 3
    assert p[:10] == bytes([5, 0, 0, 0, 0xCA, 0x08, 3, 9, 2, 0])       # index, 2250, 3 leads, 9 s, 2 beats
    assert np.frombuffer(p[10:10 + 6], '<i2').tolist() == [0, 1, 2]       # interleaved by sample


def test_c_stage_reference_matches_the_model_layers():
    """mcu_export.c_stage (what App/ecg_stage.c computes) == QRSTCancel / RhythmDescriptor."""
    import tensorflow as tf
    from ecgr.models.layers import RhythmDescriptor
    from ecgr.rhythm import mcu_export
    from ecgr.rhythm.dualunet import BEAT_HZ, QRSTCancel
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2500, 3)).astype(np.float32)
    p = np.clip(rng.random(1250) ** 6 * 1.2, 0, 1).astype(np.float32)
    qc, rd = QRSTCancel(), RhythmDescriptor(step_hz=BEAT_HZ, channel=0)
    rd.build((None, 1250, 1))
    res, qrs = qc([tf.constant(x[None]), tf.constant(p[None, :, None])])
    rr = rd(tf.constant(p[None, :, None]))

    class _M:                                       # stage_constants reads two layers
        def get_layer(self, name):
            return {'qrst_cancel': qc, 'rr_ac': rd}[name]
    k = mcu_export.stage_constants(_M())
    assert (k['pre'], k['post'], k['nms'], k['mask'], k['rr_lo'], k['rr_hi']) == (12, 56, 51, 17, 34, 250)
    r2, q2, rr2 = mcu_export.c_stage(x, p, k)
    assert np.abs(np.asarray(res)[0] - r2).max() < 1e-5
    assert np.array_equal(np.asarray(qrs)[0, :, 0], q2) and q2.sum() > 0
    assert np.abs(np.asarray(rr)[0] - rr2).max() < 1e-6


def test_split_export_reproduces_a_fresh_30k_model():
    from ecgr.rhythm import mcu_export
    from ecgr.rhythm.dualunet import SIZES, build_dual_unet_model
    model = build_dual_unet_model(**SIZES['rhythm_dual_30k'], name='dualunet_rhythm_30k')
    ex = mcu_export.export(model)
    assert [n for n, _ in ex['b_inputs']] == list(mcu_export.B_INPUTS)
    diff = mcu_export.check(model, ex, n=1)
    assert max(diff.values()) < 1e-4, diff


# --- the virtual kit -------------------------------------------------------------------------

def _synthetic_source(seconds=46):
    """3-lead 250 Hz test ECG (lead 3 missing, as on mitdb) with AFIB labels on seconds 20-29."""
    t = np.arange(seconds * 250) / 250
    rng = np.random.default_rng(0)
    r = np.cumsum(0.75 + 0.1 * rng.standard_normal(int(seconds / 0.7)))
    sig = np.zeros((len(t), 3))
    for tr in r[r < seconds]:
        sig[:, 0] += 1.2 * np.exp(-0.5 * ((t - tr) / 0.012) ** 2) + 0.3 * np.exp(-0.5 * ((t - tr - 0.25) / 0.04) ** 2)
        sig[:, 1] -= 0.6 * np.exp(-0.5 * ((t - tr) / 0.015) ** 2)
    sig[:, :2] += 0.02 * rng.standard_normal((len(t), 2))
    src = sl.RecordSource.__new__(sl.RecordSource)
    src.uv = np.round(sig * 1000).astype(np.int16)
    src.start, src.seconds = 0, seconds
    src.labels = np.zeros(seconds, np.uint8)
    src.labels[20:30] = 1
    pos = np.round(r[r < seconds - 0.1] * 250).astype(int)
    src.beats = pos, np.ones(len(pos), int)
    src.has_rhythm = src.has_beats = True
    return src


def _python_reference(src, n):
    """What the kit should answer: TFLite nets + c_stage, per-second means, overlap = mean of
    both windows, beats from ecgr.rhythm.beats.pick_beats."""
    from ecgr.rhythm import mcu_export
    from ecgr.rhythm.beats import pick_beats
    meta = json.load(open(os.path.join(MODEL_DIR, 'model_meta.json')))
    ex = dict(a=open(os.path.join(MODEL_DIR, 'ecga.tflite'), 'rb').read(),
              b=open(os.path.join(MODEL_DIR, 'ecgb.tflite'), 'rb').read(), constants=meta['constants'])
    out, prev = [], None
    for k in range(n):
        w = src.uv[2250 * k:2250 * k + 2500].astype(np.float64) * 1e-3
        sd = w.std(0)
        x = ((w - w.mean(0)) / np.where(sd > 0, sd, 1)).astype(np.float32)
        y = mcu_export.run_split(ex, x)
        sec, beat = y['rhythm'].reshape(10, 125, 5).mean(1), y['beat'].copy()
        tail = sec[9].copy(), y['beat'][-125:].copy()
        if prev is not None:
            sec[0] = (sec[0] + prev[0]) / 2
            beat[:125] = (beat[:125] + prev[1]) / 2
        prev = tail
        b = pick_beats(beat, 125)
        out.append(dict(sec=sec, beats=[(int(round(t * 250)), int(c)) for t, c in zip(b['t'], b['cls'])]))
    return out


@pytest.mark.skipif(not (os.path.exists(sl.SIM_BIN) and os.path.exists(os.path.join(MODEL_DIR, 'ecga.tflite'))),
                    reason='virtual kit not built (export_model.py + make -C firmware/stm32f746g_disco/host)')
def test_virtual_kit_answers_like_the_python_pipeline(tmp_path):
    src = _synthetic_source()
    n = src.n_chunks()
    assert n == 5                                   # 10 s + 4 x 9 s of 46 s
    vk = sl.VirtualKit(str(tmp_path / 'lcd.ppm'))
    sess = sl.KitSession(vk.start(), 921600, src, pace='fast', virtual=vk).start()
    sess.thread.join(120)
    snap = sess.snapshot()
    vk.stop()
    assert snap['status'] == 'done', snap
    assert snap['info']['virtual'] and snap['info']['window'] == 2500 and snap['info']['hop'] == 2250
    ref = _python_reference(src, n)
    for k, (r, e) in enumerate(zip(snap['results'], ref)):
        assert r['index'] == k and r['win_start'] == 2250 * k
        assert np.abs(np.array(r['prob']) - e['sec']).max() <= 1 / 255 + 1e-6, k   # u8 on the wire
        assert r['pred'] == e['sec'].argmax(1).tolist(), k
        assert [(b['pos'], b['cls']) for b in r['beats']] == e['beats'], k
        assert r['ref'] == src.labels[9 * k:9 * k + 10].tolist()                       # labels echoed
    head = open(tmp_path / 'lcd.ppm', 'rb').read(15)
    assert head.startswith(b'P6\n480 272\n255\n')                                     # LCD frame written


def test_kit_refuses_a_chunk_out_of_sequence(tmp_path):
    if not os.path.exists(sl.SIM_BIN):
        pytest.skip('virtual kit not built')
    import serial
    vk = sl.VirtualKit()
    ser = serial.Serial(vk.start(), 921600, timeout=0.1)
    try:
        src = _synthetic_source()
        reader, frames = sl.FrameReader(), []
        ser.write(sl.encode(sl.HELLO, 0) + sl.encode(sl.CHUNK, 1, src.chunk(1)))   # index 1 first
        for _ in range(50):
            frames += reader.feed(ser.read(4096))
            if len(frames) >= 2:
                break
        assert [f[0] for f in frames[:2]] == [sl.INFO, sl.ERROR]
        err = sl.parse_error(frames[1][2])
        assert err['kind'] == 'sequence' and err['index'] == 1
    finally:
        ser.close()
        vk.stop()


@pytest.mark.skipif(not os.path.exists(os.path.join(MITDB, '201.hea')), reason='no mitdb')
def test_record_source_cuts_mitdb_into_kit_chunks():
    src = sl.RecordSource(os.path.join(MITDB, '201'), start_s=0)
    assert src.uv.shape[1] == 3 and not src.uv[:, 2].any()          # 2 leads + a zero lead
    assert src.chunk_bounds(0) == (0, 2500) and src.chunk_bounds(2) == (4750, 7000)
    assert len(src.chunk(0)) > 15000 and len(src.chunk(1)) < len(src.chunk(0))
    assert src.has_rhythm and set(np.unique(src.labels)) <= {0, 1, 2, 3, 4}
    assert (src.labels == 1).sum() > 0                              # 201 has AFIB
    w = src.window(1)
    assert w['start'] == 2250 and w['uv'].shape == (2500, 3) and len(w['labels']) == 10


def test_kit_tab_and_api_without_a_session(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    import app
    (tmp_path / 'mydb').mkdir()
    (tmp_path / 'mydb' / 'r1.hea').write_text('r1 1 250 2500\n')

    async def go():
        client = TestClient(TestServer(app.make_app(str(tmp_path), str(tmp_path), 'none')))
        await client.start_server()
        try:
            cfg = await (await client.get('/api/kit/config')).json()
            recs = await (await client.get('/api/kit/records', params={'db': 'mydb'})).json()
            state = await (await client.get('/api/kit/state')).json()
            win = await client.get('/api/kit/window', params={'index': 0})
            page = await (await client.get('/')).text()
            return cfg, recs, state, win.status, page
        finally:
            await client.close()

    cfg, recs, state, win_status, page = asyncio.run(go())
    assert cfg['dbs'] == ['mydb'] and cfg['baud'] == app.KIT_BAUD
    assert recs['records'] == ['r1'] and state['status'] == 'idle' and win_status == 404
    assert 'id="tk"' in page and 'function kResult' in page
