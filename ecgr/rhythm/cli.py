"""Entry point of the rhythm task.

    python -m ecgr.rhythm config
    python -m ecgr.rhythm models
    python -m ecgr.rhythm data     [--splits train eval test] [--sources ...] [--limit N]
    python -m ecgr.rhythm audit                            # re-prove the holdout from disk
    python -m ecgr.rhythm train    --model rhythm_1m [--epochs N] [--backbone-from beat.keras]
    python -m ecgr.rhythm eval     --checkpoint best_model.keras [--split test] [--snrs ...]
    python -m ecgr.rhythm predict  --checkpoint best_model.keras --record path/to/rec [...]
    python -m ecgr.rhythm ec57     --checkpoint best_model.keras --tag mytag [...]
    python -m ecgr.rhythm ec57     --tag mytag --decode-only [--sweep]   # from the stored npz
    python -m ecgr.rhythm xai      --checkpoint best_model.keras [--explain rec:second[:CLASS]]
    python -m ecgr.rhythm xai-web  --checkpoint best_model.keras [--port 8777] [--ec57-out DIR]
"""
import argparse
import sys


def build_parser():
    from . import config as rc
    from .model import list_models
    p = argparse.ArgumentParser(prog='ecgr.rhythm', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='stage', required=True)
    sub.add_parser('config', help='print the resolved configuration')
    sub.add_parser('models', help='the model family with parameter counts')

    d = sub.add_parser('data', help='build the train / eval / test windows')
    d.add_argument('--splits', nargs='*', default=['train', 'eval', 'test'],
                   choices=['train', 'eval', 'test'])
    d.add_argument('--sources', nargs='*', default=None, choices=list(rc.TRAIN_SOURCES))
    d.add_argument('--limit', type=int, default=None, help='first N events per split')
    d.add_argument('--workers', type=int, default=None)
    d.add_argument('--no-audit', action='store_true')

    sub.add_parser('audit', help='re-check train/eval/test study separation on disk')

    t = sub.add_parser('train', help='train one model')
    t.add_argument('--model', choices=list_models(), required=True)
    t.add_argument('--epochs', type=int, default=None)
    t.add_argument('--batch-size', type=int, default=None)
    t.add_argument('--lr', type=float, default=None)
    t.add_argument('--patience', type=int, default=None)
    t.add_argument('--steps-per-epoch', type=int, default=None)
    t.add_argument('--max-windows', type=int, default=None, help='smoke test')
    t.add_argument('--init-from', default=None, help='all weights from a rhythm checkpoint')
    t.add_argument('--backbone-from', default=None,
                   help='backbone from a BEAT checkpoint (.keras) or ssl_backbone.weights.h5 '
                        'of the same size')
    t.add_argument('--init-matching', default=None,
                   help='warm start: every layer with the same name and shapes in this '
                        'rhythm checkpoint takes its weights (e.g. rhythm_unet1250_1m -> '
                        'rhythm_unet1250b_1m: encoder, rhythm decoder, noise head)')
    t.add_argument('--freeze-epochs', type=int, default=0,
                   help='with --init-matching: epochs training only the new layers at --lr '
                        'before everything trains at --lr2')
    t.add_argument('--lr2', type=float, default=None, help='phase-2 learning rate (lr / 5)')
    t.add_argument('--sampler', choices=['uniform', 'stratified'], default=None,
                   help=f'batch composition (default rc.SAMPLER = {rc.SAMPLER})')
    t.add_argument('--schedule', choices=['plateau', 'cosine'], default=None,
                   help=f'learning-rate schedule (default rc.LR_SCHEDULE = {rc.LR_SCHEDULE})')

    e = sub.add_parser('eval', help='score a checkpoint (default: the rhythm test set)')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--split', default='test', choices=['train', 'eval', 'test'])
    e.add_argument('--snrs', nargs='*', default=None,
                   help="noise levels in dB; 'clean' = the windows as stored "
                        "(default: clean 18 12 6 0 -6)")
    e.add_argument('--batch-size', type=int, default=None)
    e.add_argument('--max-windows', type=int, default=None)

    r = sub.add_parser('predict', help='predict whole WFDB records')
    r.add_argument('--checkpoint', required=True)
    r.add_argument('--record', nargs='+', required=True, help='record paths, no extension')
    r.add_argument('--out', default='rhythm_predictions')

    x = sub.add_parser('ec57', help='score a checkpoint with epicmp against real WFDB '
                                    'rhythm annotations (Physionet + the rhythm holdout)')
    x.add_argument('--checkpoint', default=None,
                   help='required unless --decode-only / --sweep (the npz already exist)')
    x.add_argument('--tag', default=None, help='output folder under rc.EC57_DIR '
                                               '(default: rc.RUN_TAG)')
    x.add_argument('--dbs', nargs='*', default=None, choices=list_ec57_dbs(),
                   help=f'Physionet databases (default {rc.EC57_DEFAULT_DBS})')
    x.add_argument('--classes', nargs='*', default=None, choices=rc.EC57_CLASSES,
                   help='rhythm classes to score (default: all five, per rc.EC57_CLASS_DBS)')
    x.add_argument('--max-records', type=int, default=None)
    x.add_argument('--decode-only', '--predict-only', dest='decode_only', action='store_true',
                   help='re-decode the stored per-second probabilities (npz) with the '
                        'current rc.DECODE_* and re-run epicmp - no inference')
    x.add_argument('--sweep', action='store_true',
                   help='after scoring, sweep the decoding parameters from the npz '
                        '(ec57.sweep_decoding) - implies nothing about inference; combine '
                        'with --decode-only to skip it')
    x.add_argument('--include-paced', dest='include_excluded', action='store_true',
                   help=f'also score the records rc.EC57_EXCLUDE_RECORDS leaves out '
                        f'({rc.EC57_EXCLUDE_RECORDS})')
    x.add_argument('--sweep-on', choices=['ec57', 'validation'], default='ec57',
                   help="cells the sweep optimises: the EC57 target table, or the held-out "
                        "non-EC57 records (rc.VALIDATION_TARGET_CELLS; use with "
                        "--dbs ltafdb nsrdb incartdb --skip-rhythm-eval)")
    x.add_argument('--no-beat-pp', dest='beat_pp', action='store_false', default=None,
                   help='decode from the step track only, skipping the beat-level rhythm '
                        'post-processing (rhythm/beats.py) even when beats are stored - '
                        'for the A/B against the model with a beat decoder')
    x.add_argument('--skip-physionet', action='store_true')

    z = sub.add_parser('xai', help='explain a dual U-Net checkpoint: layer probes, layer '
                                   'statistics, inference-time sweeps, per-second attributions')
    z.add_argument('--checkpoint', required=True)
    z.add_argument('--out', default=None, help='report folder (default <run>/xai/<checkpoint stem>)')
    z.add_argument('--parts', nargs='*', default=['probe', 'layers', 'experiment', 'explain'],
                   choices=['probe', 'layers', 'experiment', 'explain'])
    z.add_argument('--split', default='eval', choices=['train', 'eval', 'test'])
    z.add_argument('--per-class', type=int, default=300, help='windows per class category')
    z.add_argument('--explain', nargs='*', default=[],
                   help="record paths with a second and optional class: "
                        "'/path/mitdb/223:1200' or '/path/mitdb/223:1200:VT'")
    z.add_argument('--explain-ec57', default=None,
                   help='an EC57 output folder (<run>/ec57/<tag>): explain the longest false '
                        'positive / negative stretch of every class on mitdb')
    z.add_argument('--explain-top', type=int, default=2)

    w = sub.add_parser('xai-web', help='interactive XAI web service: edit the input, apply layer '
                                       'interventions, see layer heatmaps and the output')
    w.add_argument('--checkpoint', required=True)
    w.add_argument('--host', default='0.0.0.0')
    w.add_argument('--port', type=int, default=8777)
    w.add_argument('--ec57-out', default=None,
                   help='an EC57 output folder (<run>/ec57/<tag>): whole-record prediction '
                        'strip and the EC57 error-source finding')
    w.add_argument('--xai-report', default=None,
                   help='xai_report.json of `xai` (error sources without re-reading EC57)')
    w.add_argument('--gpu', action='store_true', help='use the GPU (default CPU: it is shared '
                                                      'with training)')
    x.add_argument('--skip-rhythm-eval', action='store_true')
    return p


def list_ec57_dbs():
    from .. import config as base_config
    from . import config as rc
    return list(base_config.EC57_DBS) + list(rc.VALIDATION_CLASS_DBS)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stage == 'xai-web' and not args.gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''          # before TensorFlow initialises the GPU
    from . import config as rc

    if args.stage == 'config':
        print(rc.describe())
    elif args.stage == 'models':
        from .model import build, list_models
        for name in list_models():
            m = build(name)
            out = {k: tuple(v.shape[1:]) for k, v in m.output.items()}
            print(f"{name:12s} {m.count_params():>10,} params  in {m.input_shape[1:]} "
                  f"out {out}")
    elif args.stage == 'data':
        from .build import build
        build(sources=args.sources, limit=args.limit, workers=args.workers,
              splits=tuple(args.splits), audit=not args.no_audit)
    elif args.stage == 'audit':
        from .build import audit_written
        audit_written()
    elif args.stage == 'train':
        from .train import train
        train(args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              patience=args.patience, steps_per_epoch=args.steps_per_epoch,
              max_windows=args.max_windows, init_from=args.init_from,
              init_matching_from=args.init_matching, freeze_epochs=args.freeze_epochs,
              lr2=args.lr2, sampler=args.sampler, schedule=args.schedule,
              backbone_from=args.backbone_from)
    elif args.stage == 'eval':
        from .evaluate import DEFAULT_SNRS, evaluate
        snrs = DEFAULT_SNRS if args.snrs is None else tuple(
            None if s == 'clean' else float(s) for s in args.snrs)
        evaluate(args.checkpoint, split=args.split, snrs=snrs, batch_size=args.batch_size,
                 max_windows=args.max_windows)
    elif args.stage == 'predict':
        import tensorflow as tf
        from .predict import predict_record
        model = tf.keras.models.load_model(args.checkpoint, compile=False)
        for path in args.record:
            rhythm, _, windows, episodes = predict_record(model, path, out_dir=args.out)
            print(f"{path}: {windows[-1]['stop'] if windows else 0} s")
            print("  windows: " + ' '.join(
                f"{w['start']}-{w['stop']}:" + (w['lead'] if 'lead' in w else
                                                 '/'.join(w['channel']) if 'channel' in w else
                                                 'p_noise=' + '/'.join(f"{v:.2f}"
                                                                       for v in w['noise']))
                for w in windows))
            for ep in episodes:
                print(f"  {ep['start']:>7g}-{ep['stop']:<7g} {ep['rhythm']:6s} "
                      f"p={ep['prob']:.2f}")
    elif args.stage == 'xai':
        import os
        from .xai import run_all
        items = []
        for e in args.explain:
            parts = e.split(':')
            cls = rc.CLASS_NAMES.index(parts[2]) if len(parts) > 2 else None
            items.append((parts[0], int(parts[1]), cls))
        if args.explain_ec57:
            from .xai import ec57_error_seconds
            items += ec57_error_seconds(args.explain_ec57, top=args.explain_top)
        out = args.out or os.path.join(rc.RUN_DIR, 'xai')
        run_all(args.checkpoint, out, parts=tuple(args.parts), split=args.split,
                per_class=args.per_class, explain_items=items, ec57_out=args.explain_ec57)
    elif args.stage == 'xai-web':
        from .xai_web import serve
        serve(args.checkpoint, host=args.host, port=args.port, ec57_out=args.ec57_out,
              xai_report=args.xai_report)
    elif args.stage == 'ec57':
        from .ec57 import run
        run(args.checkpoint, args.tag or rc.RUN_TAG, dbs=args.dbs, classes=args.classes,
            max_records=args.max_records, decode_only=args.decode_only,
            skip_physionet=args.skip_physionet, skip_rhythm_eval=args.skip_rhythm_eval,
            sweep=args.sweep, include_excluded=args.include_excluded,
            sweep_on=args.sweep_on,
            decode=None if args.beat_pp is None else {'beat_pp': args.beat_pp})
    return 0


if __name__ == '__main__':
    sys.exit(main())
