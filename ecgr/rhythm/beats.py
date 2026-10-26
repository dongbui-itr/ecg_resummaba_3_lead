"""Beats from the beat decoder, and rhythm post-processing on the beat train.

The production Holter pipeline runs its second-stage rhythm logic per BEAT: every beat carries
one rhythm class, contiguous same-class regions are validated against criteria (duration,
beat count, rate, runs of V or S beats), invalid regions are merged into their neighbours, VT
and SVT are stretched along V / S runs and cut where the beats do not match. This module is
that stage for the models with a 'beat' output, with the defects the analysis of the
production code lists fixed (config.BEAT_PP_* and the comments below).

    pick_beats       (T, 4) none/N/S/V probabilities at step_hz -> {t, cls, conf, probs}
    postprocess      step class track + beats -> corrected track, beat symbols (S in AF -> N)

The step track comes from labels.decode_track (smoothing, prior correction, minimums, the
AF confidence floor); the beat stage runs after it and only changes the stretch the beats
cover. A record with fewer than 3 beats is returned unchanged.

Order of the stage (production order, with the fixes):
    1. rhythm per beat (majority of the track in the beat's cell)
    2. runs of V / S beats that meet the VT / SVT criteria become VT / SVT on their own -
       the production VES_RUN / SVES_RUN events that the EC57 export counts as VT (K);
       off unless options['runs_to_rhythm'][class] (chosen on validation)
    3. a V (S) run with any VT (SVT) beat is VT (SVT) throughout
    4. criteria per region (A: beat counts enforced, J: VT / SVT need a fast rate)
    5. a maximal invalid stretch - the gaps between invalid regions tested on the beats, not
       on the valid mask (C) - of >= long_invalid seconds becomes SINUS (H: absolute seconds)
    6. the remaining invalid regions take their neighbours by an explicit rule (E)
    7. VT / SVT cut at non-matching beats, which take the surrounding rhythm (D)
    8. optional AFib / SVT window arbitration (B, G fixed)
    9. beat symbols: S inside AFIB -> N; optional prematurity gate on S outside SVT
"""
import numpy as np

from . import config as rc
from .labels import NOISE, class_index, runs

N_BEAT, S_BEAT, V_BEAT = 1, 2, 3


def pick_beats(beat_probs, step_hz, threshold=None, refractory_s=None):
    """Local maxima of p(beat) = 1 - p(none) above `threshold`, at least `refractory_s`
    apart (the higher peak wins). cls = argmax of N/S/V around the peak (+-2 steps)."""
    threshold = rc.BEAT_PICK_THRESHOLD if threshold is None else threshold
    refractory_s = rc.BEAT_REFRACTORY_SECONDS if refractory_s is None else refractory_s
    probs = np.asarray(beat_probs, dtype=np.float32)
    empty = dict(t=np.zeros(0), cls=np.zeros(0, int), conf=np.zeros(0, np.float32),
                 probs=np.zeros((0, 3), np.float32))
    if len(probs) < 3:
        return empty
    p = 1.0 - probs[:, 0]
    ps = np.convolve(p, np.ones(3, np.float32) / 3.0, mode='same')
    inner = ps[1:-1]
    cand = np.flatnonzero((inner >= ps[:-2]) & (inner > ps[2:]) & (inner > threshold)) + 1
    if len(cand) == 0:
        return empty
    r = max(1, int(round(refractory_s * step_hz)))
    taken = np.zeros(len(ps), dtype=bool)
    keep = []
    for i in cand[np.argsort(-ps[cand], kind='stable')]:
        if not taken[i]:
            keep.append(i)
            taken[max(0, i - r):i + r + 1] = True
    keep = np.array(sorted(keep), dtype=np.int64)
    win = np.stack([probs[max(0, i - 2):i + 3, 1:4].mean(axis=0) for i in keep])
    return dict(t=keep / float(step_hz), cls=1 + np.argmax(win, axis=1),
                conf=ps[keep].astype(np.float32), probs=win.astype(np.float32))


# [VI] Mỗi beat sở hữu một "ô" trên track step: từ trung điểm với beat trước đến trung điểm với
# beat sau (ô đầu/cuối kéo tới hai đầu record). Đây là cách export sản phẩm đặt mốc chuyển
# rhythm giữa hai beat.
# [VI] Lọc beat V lưỡng lự: beat đang mang nhãn V mà p(V) không vượt max(p(N), p(S)) ít
# nhất `margin` thì đổi sang lớp lớn hơn trong N/S. Dùng xác suất beat đã lưu trong npz nên
# không cần chạy lại model. margin 0 = giữ nguyên argmax.
def relabel_v(beats, margin):
    """Beats whose V call wins by less than `margin` (p(V) - max(p(N), p(S)), from the stored
    per-beat probabilities) become the larger of N / S. Returns a new dict; margin <= 0 or
    no beats = unchanged. Used by ec57.decode_record (decode['v_margin'])."""
    if beats is None or not margin or margin <= 0 or len(beats['t']) == 0:
        return beats
    probs = np.asarray(beats['probs'], dtype=np.float32)          # (n, 3) N, S, V
    cls = np.asarray(beats['cls'], dtype=int).copy()
    weak = (cls == V_BEAT) & (probs[:, 2] - probs[:, :2].max(axis=1) < margin)
    cls[weak] = 1 + np.argmax(probs[weak, :2], axis=1)
    return dict(beats, cls=cls)


def relabel_s(beats, margin):
    """The S twin of relabel_v: beats whose S call wins by less than `margin` (p(S) -
    max(p(N), p(V))) become the larger of N / V. Used by ec57.decode_record
    (decode['s_margin']) - the SVEB +P lever of the validation tuning."""
    if beats is None or not margin or margin <= 0 or len(beats['t']) == 0:
        return beats
    probs = np.asarray(beats['probs'], dtype=np.float32)          # (n, 3) N, S, V
    cls = np.asarray(beats['cls'], dtype=int).copy()
    weak = (cls == S_BEAT) & (probs[:, 1] - probs[:, [0, 2]].max(axis=1) < margin)
    cls[weak] = np.where(probs[weak, 0] >= probs[weak, 2], N_BEAT, V_BEAT)
    return dict(beats, cls=cls)


def beat_cells(t, n_steps, step_hz):
    """Step range [a, b) each beat owns: midpoints to its neighbours (the production export
    puts rhythm transitions at midpoints between beats); the first and last cell run to the
    record's ends so the beat rhythm covers the whole track."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    mids = np.round((t[1:] + t[:-1]) / 2.0 * step_hz).astype(int)
    mids = np.clip(mids, 0, n_steps)
    a = np.concatenate([[0], mids])
    b = np.concatenate([mids, [n_steps]])
    return a, b


# [VI] Rhythm của một beat = lớp chiếm đa số trong ô của nó (ô rỗng -> SINUS). Từ đây mọi bước
# gộp/tách của tầng 2 làm việc trên mảng r (một rhythm cho mỗi beat), không còn trên step.
def rhythm_per_beat(track, a, b):
    """Majority class of the track inside each beat's cell (SINUS for an empty cell)."""
    out = np.full(len(a), rc.SINUS, dtype=int)
    for i, (lo, hi) in enumerate(zip(a, b)):
        if hi > lo:
            out[i] = int(np.bincount(track[lo:hi], minlength=NOISE + 1).argmax())
    return out


# --- criteria ---------------------------------------------------------------------------------

def _duration(t, lo, hi):
    return float(t[hi - 1] - t[lo]) if hi - lo >= 2 else 0.0


def _hr(t, lo, hi):
    if hi - lo < 2:
        return float('nan')
    rr = np.diff(t[lo:hi])
    return 60.0 / float(rr.mean()) if rr.mean() > 0 else float('nan')


def _max_run(sym, lo, hi, value):
    best = cur = 0
    for v in sym[lo:hi]:
        cur = cur + 1 if v == value else 0
        best = max(best, cur)
    return best


# [VI] Kiểm tra một vùng beat liên tục [lo, hi) mang rhythm `name` có đạt tiêu chí không
# (rc.BEAT_PP_CRITERIA): duration, num_beat (lỗi A của sản phẩm đã sửa: có tác dụng thật),
# min_hr cho VT/SVT (lỗi J), max_hr cho AVB, run V liên tiếp cho VT. SVT được chấp nhận khi
# có run S, HOẶC tỉ lệ S >= min_frac, HOẶC HR nhảy vọt >= onset_ratio x HR 4 beat trước.
# Hàm này không gộp gì; kết quả valid/invalid quyết định hai bước gộp bên dưới.
def region_valid(name, t, sym, lo, hi, prev_hr=None, criteria=None):
    """Does the region of beats [lo, hi) called `name` meet its criteria? Fixes A (beat
    counts enforced) and J (VT / SVT need a fast rate). SVT accepts N beats: in sustained
    SVT the beats lose their prematurity and are labelled N, so a run of S, a fraction of S
    or an abrupt rate jump against the preceding beats is enough (fix D's companion)."""
    criteria = rc.BEAT_PP_CRITERIA if criteria is None else criteria
    c = criteria.get(name)
    if c is None:
        return True
    n = hi - lo
    dur, hr = _duration(t, lo, hi), _hr(t, lo, hi)
    if 'duration' in c and dur < c['duration']:
        return False
    if 'num_beat' in c and n < c['num_beat']:
        return False
    if 'min_hr' in c and not hr >= c['min_hr']:
        return False
    if 'max_hr' in c and not hr <= c['max_hr']:
        return False
    if name == 'VT':
        return _max_run(sym, lo, hi, V_BEAT) >= c['run']
    if name == 'SVT':
        if _max_run(sym, lo, hi, S_BEAT) >= c['run']:
            return True
        if np.mean(sym[lo:hi] == S_BEAT) >= c['min_frac']:
            return True
        return prev_hr is not None and prev_hr > 0 and hr >= c['onset_ratio'] * prev_hr
    return True


# --- options ---------------------------------------------------------------------------------

def default_options():
    """The tunable switches of the stage, from config. `postprocess(options=...)` overrides
    any of them (the validation grid in tune.py passes them explicitly)."""
    return dict(
        long_invalid=rc.BEAT_PP_LONG_INVALID_SECONDS,
        fast_rr=rc.BEAT_PP_FAST_RR_SECONDS,
        runs_to_rhythm=dict(rc.BEAT_PP_RUNS_TO_RHYTHM),
        afib_svt_merge=rc.BEAT_PP_AFIB_SVT_MERGE,
        svt_ratio=rc.BEAT_PP_SVT_RATIO,
        s_prematurity=rc.BEAT_PP_S_PREMATURITY,
        s_in_afib_to_n=rc.BEAT_PP_S_IN_AFIB_TO_N,
    )


def _options(options):
    out = default_options()
    for k, v in (options or {}).items():
        if k == 'runs_to_rhythm':
            out[k] = {**out[k], **(v or {})}
        else:
            out[k] = v
    return out


# --- the stage -----------------------------------------------------------------------------

def _spec():
    return {class_index(n) for n in ('VT', 'SVT', 'AVB')}


def _rank():
    order = {}
    for i, name in enumerate(reversed(rc.BEAT_PP_PRIORITY)):
        order[NOISE if name == 'NOISE' else class_index(name)] = i
    return order


# [VI] Bước 2 (tuỳ chọn, mặc định tắt; bản tuned bật): một chuỗi beat V liên tiếp tự thoả tiêu
# chí VT (số beat, run, HR) thì cả chuỗi thành VT dù track step không nói; tương tự S -> SVT.
# Tương đương VES_RUN / SVES_RUN của sản phẩm mà export EC57 cộng vào VT. Bỏ qua beat trong NOISE.
def _runs_to_rhythm(r, t, sym, criteria, which):
    """Step 2: a run of V (S) beats that on its own meets the VT (SVT) criteria - length,
    rate, beat count - is VT (SVT) even where the step track never said so. This is the
    production VES_RUN / SVES_RUN beat event, which the EC57 export folds into VT (K); the
    reference VT episodes of mitdb have a median length of 1.8 s, which a 10 s rhythm head
    smooths over more often than the beat decoder misses the V beats. Beats in NOISE cells
    are left alone."""
    for name, value in (('VT', V_BEAT), ('SVT', S_BEAT)):
        if not which.get(name):
            continue
        c = class_index(name)
        for lo, hi in runs((sym == value).astype(int)):
            if sym[lo] != value or np.any(r[lo:hi] == NOISE):
                continue
            if region_valid(name, t, sym, lo, hi, None, criteria):
                r[lo:hi] = c
    return r


# [VI] Bước 3 (giống sản phẩm): chuỗi V liên tiếp có BẤT KỲ beat nào mang VT -> cả chuỗi VT.
# Tương tự chuỗi S với SVT.
def _extend_along_runs(r, sym):
    """Step 3: a V (S) run with any VT (SVT) beat is VT (SVT) throughout."""
    for value, name in ((V_BEAT, 'VT'), (S_BEAT, 'SVT')):
        c = class_index(name)
        for lo, hi in runs((sym == value).astype(int)):
            if sym[lo] == value and np.any(r[lo:hi] == c):
                r[lo:hi] = c
    return r


# [VI] Bước 4: gọi region_valid cho từng vùng liên tục cùng rhythm, trả về mảng valid theo beat.
# prev_hr = HR của tối đa 4 beat ngay trước vùng, dùng cho tiêu chí onset của SVT.
def _validate(r, t, sym, criteria):
    valid = np.ones(len(r), dtype=bool)
    for lo, hi in runs(r):
        c = int(r[lo])
        name = 'NOISE' if c == NOISE else rc.CLASS_NAMES[c]
        prev_hr = _hr(t, max(0, lo - 4), lo) if lo >= 2 else None
        valid[lo:hi] = region_valid(name, t, sym, lo, hi, prev_hr, criteria)
    return valid


# [VI] Gộp bước 5 (sửa lỗi C, H): một dải invalid LIÊN TỤC TỐI ĐA (bất kể gồm những lớp nào)
# dài >= long_invalid giây (mặc định 10 s) -> SINUS và được đánh dấu valid. Thay cho ngưỡng
# "1/4 record" của sản phẩm (phụ thuộc độ dài record) và cách xét gap sai trên mask valid.
def _long_invalid_to_sinus(r, t, valid, long_invalid):
    """Step 5 (fixes C and H): the production step joins invalid VT / SVT / AVB regions
    across gaps and turns the joined stretch into SINUS when it is long enough - but tests
    the gap on the valid mask instead of the rhythm, so it joins across any invalid beats.
    Done properly, "invalid region, invalid gap, invalid region" is simply one maximal
    invalid stretch, whatever the classes: that is what is measured here, in seconds."""
    for lo, hi in runs(valid.astype(int)):
        if not valid[lo] and _duration(t, lo, hi) >= long_invalid:
            r[lo:hi] = rc.SINUS
            valid[lo:hi] = True
    return r, valid


# [VI] Gộp bước 6 (sửa lỗi E, H): các vùng invalid còn lại nhận rhythm của hàng xóm theo luật
# tường minh (spec = VT/SVT/AVB):
#   - không có hàng xóm nào                 -> SINUS
#   - chỉ có một bên                        -> bên đó, nhưng nếu bên đó là spec -> SINUS
#   - hai bên giống nhau                    -> lớp đó
#   - hai bên đều là spec                   -> SINUS
#   - một bên là spec                       -> lấy bên kia
#   - còn lại                               -> lớp cao hơn theo rc.BEAT_PP_PRIORITY
#                                              (AFIB > SINUS > NOISE)
# Vùng invalid dài >= long_invalid vẫn về SINUS như bước 5.
def _merge_invalid(r, t, valid, long_invalid):
    """Step 6 (fixes E and H): invalid regions take their neighbours' rhythm by an explicit
    rule - both neighbours alike -> that; both 'spec' (VT/SVT/AVB) -> SINUS; one spec -> the
    other side; otherwise the higher of BEAT_PP_PRIORITY; a long invalid stretch -> SINUS."""
    spec, rank = _spec(), _rank()
    K = len(r)
    for lo, hi in runs(r):
        if valid[lo]:
            continue
        if _duration(t, lo, hi) >= long_invalid:
            r[lo:hi] = rc.SINUS
            continue
        pres = int(r[lo - 1]) if lo > 0 else None
        post = int(r[hi]) if hi < K else None
        if pres is None and post is None:
            new = rc.SINUS
        elif pres is None:
            new = post if post not in spec else rc.SINUS
        elif post is None:
            new = pres if pres not in spec else rc.SINUS
        elif pres == post:
            new = pres
        elif pres in spec and post in spec:
            new = rc.SINUS
        elif pres in spec:
            new = post
        elif post in spec:
            new = pres
        else:
            new = max((pres, post), key=lambda c: rank.get(c, -1))
        r[lo:hi] = new
    return r


# [VI] Rhythm "nền" quanh một vùng VT/SVT: lớp không-spec, không-NOISE ở ngay bên trái, nếu
# không có thì bên phải, mặc định SINUS. Dùng cho bước tách bên dưới.
def _background(r, lo, hi):
    """The non-spec rhythm around a VT/SVT region (left first), SINUS by default."""
    spec = _spec()
    for j in (lo - 1, hi):
        if 0 <= j < len(r) and r[j] not in spec and r[j] != NOISE:
            return int(r[j])
    return rc.SINUS


# [VI] Tách bước 7 (sửa lỗi D): trong vùng VT, beat KHÔNG phải V nhận rhythm nền xung quanh
# (AFIB giữ AFIB) thay vì bị ép SINUS như sản phẩm. Trong SVT, beat N có RR <= fast_rr (0.6 s)
# được giữ lại vì SVT kéo dài làm beat mất tính sớm. Các mảnh VT/SVT còn lại được validate lại,
# mảnh không đạt tiêu chí cũng về rhythm nền.
def _split_by_beats(r, t, sym, criteria, fast_rr):
    """Step 7 (fix D): inside VT, beats that are not V take the surrounding rhythm (not
    SINUS); inside SVT, beats that are neither S nor fast do. The fragments are validated
    again."""
    for name, value in (('VT', V_BEAT), ('SVT', S_BEAT)):
        c = class_index(name)
        for lo, hi in runs(r):
            if r[lo] != c:
                continue
            bg = _background(r, lo, hi)
            for i in range(lo, hi):
                if sym[i] == value:
                    continue
                if name == 'SVT' and i > 0 and t[i] - t[i - 1] <= fast_rr:
                    continue
                r[i] = bg
            for flo, fhi in runs(r[lo:hi]):
                flo, fhi = flo + lo, fhi + lo
                if r[flo] == c and not region_valid(name, t, sym, flo, fhi, None, criteria):
                    r[flo:fhi] = bg
    return r


# [VI] Gộp bước 8 (tuỳ chọn, mặc định tắt theo validation; sửa lỗi B, G): cửa sổ gồm các vùng
# liên tiếp chỉ có AFIB / SVT / SINUS ngắn hơn SINUS.duration, và có cả AFIB lẫn SVT, thì cả
# cửa sổ thành SVT khi thời lượng SVT >= svt_ratio x thời lượng AFIB, ngược lại thành AFIB.
# Không nuốt OTHER/VT/AVB như sản phẩm, và không còn off-by-one ở phía phải.
def _afib_svt_windows(r, t, sym, criteria, svt_ratio):
    """Step 8: the production AFib/SVT arbitration with fixes B and G: a window of
    consecutive AFIB / SVT / short-SINUS regions (nothing else) becomes SVT when SVT time >=
    ratio x AF time, else AFIB."""
    af, svt = class_index('AFIB'), class_index('SVT')
    sinus_min = criteria.get('SINUS', {}).get('duration', 0.0)
    regs = runs(r)

    def member(i):
        lo, hi = regs[i]
        c = r[lo]
        return c in (af, svt) or (c == rc.SINUS and _duration(t, lo, hi) < sinus_min)
    i = 0
    while i < len(regs):
        if not member(i):
            i += 1
            continue
        j = i
        while j + 1 < len(regs) and member(j + 1):
            j += 1
        classes = {int(r[regs[k][0]]) for k in range(i, j + 1)}
        if af in classes and svt in classes:
            dur = {af: 0.0, svt: 0.0}
            for k in range(i, j + 1):
                lo, hi = regs[k]
                if r[lo] in dur:
                    dur[int(r[lo])] += _duration(t, lo, hi)
            new = svt if dur[svt] >= svt_ratio * dur[af] else af
            r[regs[i][0]:regs[j][1]] = new
        i = j + 1
    return r


# [VI] Bước 9: ký hiệu beat ghi cho bxb. S trong AFIB -> N (reference không gán S trong AF).
# Với s_prematurity > 0: beat S ngoài SVT mà RR không ngắn hơn s_prematurity x trung vị 8 RR
# trước đó thì không "sớm" -> N. Bên trong SVT giữ nguyên.
def beat_symbols(r, t, sym, s_prematurity=0.0, s_in_afib_to_n=True):
    """Step 9: the beat labels written for bxb. S inside AFIB -> N (the reference databases
    do not label supraventricular beats inside AF; the production `S in AFIB -> N`). With
    `s_prematurity` > 0, an S beat outside SVT whose R-R is not shorter than that fraction
    of the median of the preceding 8 intervals is not premature and becomes N - inside SVT
    the beats are expected to lose their prematurity, so they are left alone."""
    out = np.asarray(sym, dtype=int).copy()
    af, svt = class_index('AFIB'), class_index('SVT')
    if s_in_afib_to_n:
        out = np.where((r == af) & (out == S_BEAT), N_BEAT, out)
    if s_prematurity and s_prematurity > 0 and len(t) > 2:
        rr = np.diff(t)
        for i in np.flatnonzero(out == S_BEAT):
            if i < 2 or r[i] == svt:
                continue
            ref = np.median(rr[max(0, i - 9):i - 1]) if i >= 3 else rr[i - 2]
            if ref > 0 and rr[i - 1] >= s_prematurity * ref:
                out[i] = N_BEAT
    return out


# [VI] Tầng 2 - post-process theo beat. Thứ tự: rhythm_per_beat -> _runs_to_rhythm ->
# _extend_along_runs -> _validate -> _long_invalid_to_sinus -> _merge_invalid -> _split_by_beats
# -> (_afib_svt_windows) -> beat_symbols -> ghi ngược: ô của mỗi beat nhận rhythm của beat,
# trừ ô đang là NOISE thì giữ NOISE. Record < 3 beat trả về nguyên trạng.
def postprocess(track, beats, step_hz, criteria=None, options=None):
    """Step class track (T,) + beats -> (track, beat symbols). See the module docstring."""
    criteria = rc.BEAT_PP_CRITERIA if criteria is None else criteria
    opt = _options(options)
    track = np.asarray(track).copy()
    t = np.asarray(beats['t'], dtype=np.float64)
    sym = np.asarray(beats['cls'], dtype=int)
    if len(t) < 3:
        return track, sym.copy()
    a, b = beat_cells(t, len(track), step_hz)
    r = rhythm_per_beat(track, a, b)

    r = _runs_to_rhythm(r, t, sym, criteria, opt['runs_to_rhythm'])
    r = _extend_along_runs(r, sym)
    valid = _validate(r, t, sym, criteria)
    r, valid = _long_invalid_to_sinus(r, t, valid, opt['long_invalid'])
    r = _merge_invalid(r, t, valid, opt['long_invalid'])
    r = _split_by_beats(r, t, sym, criteria, opt['fast_rr'])
    if opt['afib_svt_merge']:
        r = _afib_svt_windows(r, t, sym, criteria, opt['svt_ratio'])

    out_sym = beat_symbols(r, t, sym, opt['s_prematurity'], opt['s_in_afib_to_n'])
    for i, (lo, hi) in enumerate(zip(a, b)):
        if hi > lo and track[lo:hi].max() != NOISE:       # noise stays noise
            track[lo:hi] = r[i]
    return track, out_sym


# --- boundary trimming ---------------------------------------------------------------------

def trim_to_beats(track, beats, step_hz, pads):
    """Shrink every run of a class in `pads` ({class name: pad seconds}) to the beats that
    carry it: from the first to the last beat inside the run (for VT the first / last V beat,
    for SVT the first / last S or fast beat when there are any, else any beat), widened by the
    pad on each side - never beyond the run. The trimmed ends take the class on that side of
    the run (NOISE excepted: SINUS). A run without a qualifying beat is left as it is.

    Why: a decoded episode reaches to the midpoints between beats, and the probabilities are
    smoothed, so it spills into the sinus rhythm around a reviewer's caliper; on the reviewed
    strips (strip_eval) every spilled step is a SINUS sample called an arrhythmia."""
    if not pads or beats is None or len(beats['t']) == 0:
        return track
    track = np.asarray(track).copy()
    t = np.asarray(beats['t'], dtype=np.float64)
    sym = np.asarray(beats['cls'], dtype=int)
    n = len(track)
    for name, pad in pads.items():
        c = class_index(name)
        for lo, hi in runs(track):
            if track[lo] != c:
                continue
            inside = (t * step_hz >= lo) & (t * step_hz < hi)
            if name == 'VT':
                q = inside & (sym == V_BEAT)
            elif name == 'SVT':
                q = inside & (sym == S_BEAT)
            else:
                q = inside
            if not q.any():
                q = inside
            if not q.any():
                continue
            a = max(lo, int(np.floor((t[q].min() - pad) * step_hz + 1e-6)))
            b = min(hi, int(np.ceil((t[q].max() + pad) * step_hz - 1e-6)) + 1)
            left = track[lo - 1] if lo > 0 else rc.SINUS
            right = track[hi] if hi < n else rc.SINUS
            left = rc.SINUS if left == NOISE else left
            right = rc.SINUS if right == NOISE else right
            if a > lo:
                track[lo:a] = left
            if b < hi:
                track[b:hi] = right
    return track
