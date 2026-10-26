"""PC side of the STM32F746G-Discovery rhythm kit: protocol, record streaming, virtual kit.

The kit (firmware/stm32f746g_disco) receives ECG over its ST-LINK virtual COM port (USART1),
keeps a 10 s window that advances 9 s per chunk (1 s overlap), runs the model, draws the
result on its LCD and sends it back. This module streams a WFDB record to it - chunk 0 = the
first 10 s, then 9 s per chunk, each with the reference labels of those seconds - and
collects the results for the web tab.

Frames (both directions): 0xA5 0x5A | type u8 | seq u8 | len u16 | payload | crc16 u16,
little-endian, CRC-16/CCITT-FALSE over type..payload (App/ecg_proto.h on the kit).

    HELLO  0x01 PC->kit  empty: reset the stream; the kit answers INFO
    CHUNK  0x02 PC->kit  u32 index, u16 n_samples (2500 | 2250), u8 n_leads (3), u8 n_sec,
                         u16 n_ref_beats, i16 samples[n_samples * 3] (interleaved, microvolts,
                         250 Hz, preprocessed as for training: build.read_leads),
                         u8 ref_rhythm[n_sec] (0..4 = SINUS AFIB SVT VT AVB, 255 = none),
                         ref_beats[n_ref_beats] {u16 sample in the chunk, u8 1 N 2 S 3 V 4 other}
    INFO   0x81 kit->PC  u8 protocol, u8 classes, u16 fs, u16 window, u16 hop, u32 MACC,
                         u32 activation bytes, u32 weight bytes, u32 cpu Hz (0 = virtual kit),
                         char model[24], char build[24]
    RESULT 0x82 kit->PC  u32 index, u32 window start (samples), u32 inference us, u32 total us,
                         u8 n_sec, n_sec x {u8 pred, u8 ref, u8 prob[5]} (prob x 255; second 0
                         of window k >= 1 = mean with window k-1), u8 p_noise[5] (per 2 s),
                         u8 lead[5], u16 heart rate x 10, u16 n_beats,
                         n_beats x {u16 sample in the window, u8 class 1..3, u8 conf x 255},
                         u32 us net A, u32 us stage, u32 us net B
    ERROR  0x83 kit->PC  u8 code, u32 chunk index, text
    LOG    0x84 kit->PC  text

The virtual kit is the same firmware code built for the PC (firmware/stm32f746g_disco/host,
`ecgkit_sim --pty`): it opens a pseudo-terminal that this module talks to exactly like the
board's COM port.
"""
import os
import struct
import subprocess
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIRMWARE = os.path.join(ROOT, 'firmware', 'stm32f746g_disco')
SIM_BIN = os.path.join(FIRMWARE, 'build', 'host', 'ecgkit_sim')

SYNC = b'\xa5\x5a'
HELLO, CHUNK, INFO, RESULT, ERROR, LOG = 0x01, 0x02, 0x81, 0x82, 0x83, 0x84
FS = 250
WIN = 2500
HOP = 2250
LEADS = 3
CLASS_NAMES = ['SINUS', 'AFIB', 'SVT', 'VT', 'AVB']
LABEL_NONE = 255
# AAMI groups of the .atr beat symbols: 1 N, 2 S, 3 V, 4 other (F, Q, paced)
BEAT_GROUP = {**dict.fromkeys('NLRej', 1), **dict.fromkeys('AaJS', 2), **dict.fromkeys('VE', 3),
              **dict.fromkeys('F/fQ', 4)}
BEAT_NAMES = {1: 'N', 2: 'S', 3: 'V', 4: 'Q'}
ERROR_NAMES = {1: 'chunk size', 2: 'sequence', 3: 'model', 4: 'unknown type'}


# --- framing ---------------------------------------------------------------------------------

def crc16(data, crc=0xFFFF):
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _crc16_np(data):
    """Same CRC, table-driven (a 15 KB chunk in pure Python per bit is slow)."""
    crc = 0xFFFF
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc


_CRC_TABLE = [crc16(bytes([i]), 0) for i in range(256)]


def encode(msg_type, seq, payload=b''):
    head = struct.pack('<BBH', msg_type, seq & 0xFF, len(payload))
    return SYNC + head + payload + struct.pack('<H', _crc16_np(head + payload))


class FrameReader:
    """Frames out of a byte stream; bad CRCs are counted and skipped."""

    def __init__(self):
        self.buf = bytearray()
        self.crc_errors = 0

    def feed(self, data):
        self.buf += data
        frames = []
        while True:
            i = self.buf.find(SYNC)
            if i < 0:
                del self.buf[:max(0, len(self.buf) - 1)]
                return frames
            del self.buf[:i]
            if len(self.buf) < 6:
                return frames
            msg_type, seq, n = struct.unpack_from('<BBH', self.buf, 2)
            if len(self.buf) < 8 + n:
                return frames
            body = bytes(self.buf[2:6 + n])
            (crc,) = struct.unpack_from('<H', self.buf, 6 + n)
            if _crc16_np(body) == crc:
                frames.append((msg_type, seq, body[4:]))
                del self.buf[:8 + n]
            else:
                self.crc_errors += 1
                del self.buf[:2]


# --- payloads --------------------------------------------------------------------------------

def pack_chunk(index, samples_uv, ref_rhythm, ref_beats):
    """samples_uv (n, 3) int16, ref_rhythm (n // 250,) uint8, ref_beats [(sample, group)]."""
    samples_uv = np.ascontiguousarray(samples_uv, dtype='<i2')
    n = samples_uv.shape[0]
    head = struct.pack('<IHBBH', index, n, samples_uv.shape[1], len(ref_rhythm), len(ref_beats))
    beats = b''.join(struct.pack('<HB', int(s), int(g)) for s, g in ref_beats)
    return head + samples_uv.tobytes() + bytes(np.asarray(ref_rhythm, np.uint8)) + beats


def parse_info(p):
    proto, classes, fs, win, hop, macc, act, weights, cpu = struct.unpack_from('<BBHHHIIII', p)
    name = p[24:48].split(b'\0')[0].decode(errors='replace')
    build = p[48:72].split(b'\0')[0].decode(errors='replace')
    return dict(protocol=proto, classes=classes, fs=fs, window=win, hop=hop, macc=macc,
                activations=act, weights=weights, cpu_hz=cpu, model=name, build=build,
                virtual=cpu == 0)


def parse_result(p):
    index, start, us_infer, us_total, n_sec = struct.unpack_from('<IIIIB', p)
    o = 17
    pred, ref, prob = [], [], []
    for _ in range(n_sec):
        pred.append(p[o])
        ref.append(p[o + 1])
        prob.append([v / 255 for v in p[o + 2:o + 7]])
        o += 7
    noise = [v / 255 for v in p[o:o + 5]]
    lead = list(p[o + 5:o + 10])
    hr_x10, n_beats = struct.unpack_from('<HH', p, o + 10)
    o += 14
    beats = []
    for _ in range(n_beats):
        pos, cls, conf = struct.unpack_from('<HBB', p, o)
        beats.append(dict(pos=pos, cls=cls, conf=conf / 255))
        o += 4
    us_a, us_stage, us_b = struct.unpack_from('<III', p, o) if len(p) >= o + 12 else (0, 0, 0)
    return dict(index=index, win_start=start, us_infer=us_infer, us_total=us_total, pred=pred,
                ref=ref, prob=prob, noise=noise, lead=lead, hr=hr_x10 / 10, beats=beats,
                us_a=us_a, us_stage=us_stage, us_b=us_b)


def parse_error(p):
    code, index = struct.unpack_from('<BI', p)
    return dict(code=code, kind=ERROR_NAMES.get(code, '?'), index=index,
                text=p[5:].decode(errors='replace'))


# --- a record as the kit receives it ----------------------------------------------------------

def rhythm_marks(base, ext='atr'):
    """[(second, class index)] from the '+' rhythm marks of an annotation file, or None when
    the record has no such marks (AFL counts as AFIB, unmapped codes as SINUS)."""
    import wfdb
    from ecgr.rhythm import config as rc
    if not os.path.exists(f'{base}.{ext}'):
        return None, None
    ann = wfdb.rdann(base, ext)
    fs = ann.fs or wfdb.rdheader(base).fs
    marks = []
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note or [''] * len(ann.sample)):
        code = (aux or '').split('\x00')[0].strip()
        if sym == '+' and code:
            name = 'AFIB' if code == '(AFL' else rc.PHYSIONET_AUX_TO_CLASS.get(code, 'SINUS')
            marks.append((s / fs, CLASS_NAMES.index(name)))
    return (marks or None), ann


class RecordSource:
    """A WFDB record cut into the kit's chunks, preprocessed exactly like the training windows
    (ecgr.rhythm.build.read_leads: 250 Hz, band-pass, 3 leads, missing ones zero), sent as
    int16 microvolts, with per-second rhythm labels and AAMI beat labels from `ann_ext`."""

    def __init__(self, base, start_s=0, ann_ext='atr'):
        from ecgr.rhythm.build import read_leads
        leads, ratio = read_leads(base)
        self.base = base
        self.uv = np.clip(np.round(leads * 1000.0), -32768, 32767).astype(np.int16)
        self.start = int(start_s) * FS
        self.seconds = len(self.uv) // FS
        marks, ann = rhythm_marks(base, ann_ext)
        self.has_rhythm = marks is not None
        self.labels = np.full(self.seconds, LABEL_NONE, np.uint8)
        if marks:
            stops = [m[0] for m in marks[1:]] + [float('inf')]
            centers = np.arange(self.seconds) + 0.5
            for (t, cls), stop in zip(marks, stops):
                self.labels[(centers >= t) & (centers < stop)] = cls
        self.beats = np.zeros(0, np.int64), np.zeros(0, np.int64)
        if ann is not None:
            keep = [(int(round(s * ratio)), BEAT_GROUP[sym])
                    for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_GROUP]
            if keep:
                pos, grp = np.array(keep).T
                self.beats = pos, grp
        self.has_beats = len(self.beats[0]) > 0

    def chunk_bounds(self, k):
        """Samples [a, b) of chunk k (k = 0: 10 s, then 9 s each)."""
        a = self.start if k == 0 else self.start + WIN + HOP * (k - 1)
        return a, a + (WIN if k == 0 else HOP)

    def n_chunks(self):
        usable = len(self.uv) - self.start
        return 0 if usable < WIN else 1 + (usable - WIN) // HOP

    def chunk(self, k):
        a, b = self.chunk_bounds(k)
        labels = self.labels[a // FS:b // FS]
        pos, grp = self.beats
        sel = (pos >= a) & (pos < b)
        beats = list(zip(pos[sel] - a, grp[sel]))
        return pack_chunk(k, self.uv[a:b], labels, beats)

    def window(self, k):
        """The 10 s the kit analyses after chunk k: samples, labels, beats in the window."""
        a = self.start + HOP * k
        pos, grp = self.beats
        sel = (pos >= a) & (pos < a + WIN)
        return dict(start=a, uv=self.uv[a:a + WIN], labels=self.labels[a // FS:a // FS + 10],
                    beats=[dict(pos=int(p - a), cls=int(g)) for p, g in zip(pos[sel], grp[sel])])


# --- virtual kit -----------------------------------------------------------------------------

class VirtualKit:
    """`ecgkit_sim --pty` (the firmware built for the PC): start() -> pty path to open."""

    def __init__(self, snap_path=None):
        self.snap_path = snap_path
        self.proc = None
        self.port = None

    @staticmethod
    def build():
        if not os.path.exists(SIM_BIN):
            r = subprocess.run(['make', '-C', os.path.join(FIRMWARE, 'host')],
                               capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError('virtual kit build failed (run python3 firmware/stm32f746g_'
                                   'disco/export_model.py first):\n' + (r.stdout + r.stderr)[-1500:])
        return SIM_BIN

    @staticmethod
    def available():
        return os.path.exists(SIM_BIN) or os.path.isdir(os.path.join(FIRMWARE, 'build', 'host_ai'))

    def start(self):
        cmd = [self.build(), '--pty'] + (['--snap', self.snap_path] if self.snap_path else [])
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        line = self.proc.stdout.readline().strip()
        if not line.startswith('PTY '):
            self.stop()
            raise RuntimeError(f'virtual kit did not start: {line!r}')
        self.port = line[4:]
        return self.port

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def list_ports():
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return []
    # Linux lists every legacy /dev/ttyS* with hwid 'n/a'; the ST-LINK VCP is a USB ttyACM
    return [dict(device=p.device, description=p.description or '', hwid=p.hwid or '')
            for p in lp.comports() if p.hwid and p.hwid != 'n/a']


# --- one streaming session --------------------------------------------------------------------

class KitSession:
    """Streams one record to the kit in a background thread; the web tab polls `snapshot()`.

    pace 'realtime': chunk k leaves 9 k s after chunk 0, as if the PC recorded the ECG live;
    'fast': the next chunk leaves as soon as the previous result is back."""

    RESULT_TIMEOUT = 30.0

    def __init__(self, port, baud, source, max_windows=None, pace='fast', virtual=None, meta=None):
        self.port, self.baud, self.src = port, int(baud), source
        self.max_windows = max_windows
        self.pace = pace
        self.virtual = virtual
        self.meta = meta or {}
        self.lock = threading.Lock()
        self.status, self.error, self.info = 'starting', None, None
        self.results, self.log = [], []
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ser = None
        self.reader = FrameReader()
        self.seq = 0

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_flag.set()
        self.thread.join(5)
        if self.virtual:
            self.virtual.stop()

    def _note(self, text):
        with self.lock:
            self.log.append(f'{time.strftime("%H:%M:%S")} {text}')
            del self.log[:-200]

    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _send(self, msg_type, payload=b''):
        self.ser.write(encode(msg_type, self.seq, payload))
        self.seq += 1

    def _wait(self, want, timeout):
        """Next frame of type `want`; LOG frames are logged, ERROR raises."""
        end = time.monotonic() + timeout
        pending = []
        while time.monotonic() < end and not self.stop_flag.is_set():
            pending += self.reader.feed(self.ser.read(self.ser.in_waiting or 1))
            while pending:
                msg_type, _, payload = pending.pop(0)
                if msg_type == want:
                    return payload
                if msg_type == LOG:
                    self._note('kit: ' + payload.decode(errors='replace'))
                elif msg_type == ERROR:
                    e = parse_error(payload)
                    raise RuntimeError(f"kit error ({e['kind']}) at chunk {e['index']}: {e['text']}")
        if self.stop_flag.is_set():
            raise InterruptedError
        raise TimeoutError(f'no answer from the kit within {timeout:.0f} s')

    def _run(self):
        import serial
        try:
            self._set(status='connecting')
            self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
            self.ser.reset_input_buffer()
            info = None
            for attempt in range(3):
                self._send(HELLO)
                try:
                    info = parse_info(self._wait(INFO, 3.0))
                    break
                except TimeoutError:
                    self._note(f'HELLO {attempt + 1}: no INFO yet')
            if info is None:
                raise TimeoutError('the kit does not answer HELLO (port, baud, firmware?)')
            self._set(info=info, status='streaming')
            self._note(f"connected: {info['model']} ({'virtual kit' if info['virtual'] else 'STM32F746G'})")
            total = self.src.n_chunks()
            if self.max_windows:
                total = min(total, int(self.max_windows))
            t0 = time.monotonic()
            for k in range(total):
                if self.stop_flag.is_set():
                    break
                if self.pace == 'realtime' and k:
                    while time.monotonic() < t0 + 9.0 * k and not self.stop_flag.is_set():
                        time.sleep(0.05)
                sent = time.monotonic()
                self._send(CHUNK, self.src.chunk(k))
                r = parse_result(self._wait(RESULT, self.RESULT_TIMEOUT))
                r['rtt_ms'] = round((time.monotonic() - sent) * 1000, 1)
                with self.lock:
                    self.results.append(r)
            self._set(status='stopped' if self.stop_flag.is_set() else 'done')
            self._note(f'{len(self.results)} windows')
        except InterruptedError:
            self._set(status='stopped')
        except Exception as e:                         # noqa: BLE001 - reported to the page
            self._set(status='error', error=f'{type(e).__name__}: {e}')
            self._note(f'error: {e}')
        finally:
            if self.ser:
                self.ser.close()
            if self.reader.crc_errors:
                self._note(f'{self.reader.crc_errors} frames with a bad CRC')

    def snapshot(self, since=0):
        with self.lock:
            return dict(status=self.status, error=self.error, info=self.info,
                        n=len(self.results), results=self.results[since:],
                        log=self.log[-30:], meta=self.meta,
                        total=min(self.src.n_chunks(), int(self.max_windows or 1e9)))

    def result(self, k):
        with self.lock:
            return self.results[k] if 0 <= k < len(self.results) else None
