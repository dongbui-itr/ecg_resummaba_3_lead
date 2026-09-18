"""One entry point for every stage.

    python -m ecgr config                     # what this run is configured to do
    python -m ecgr models                     # the model family and its parameter counts
    python -m ecgr data      [--step npy|tfrecord|all] [--db ...] [--limit N] [--workers N]
    python -m ecgr ssl       --model resumamba_30k   # self-supervised backbone
    python -m ecgr cpc       --model resumamba_30k   # self-supervised context encoder
    python -m ecgr train     --model resumamba_30k [--epochs N] [--lr 7e-4] ...
    python -m ecgr stepeval  --model resumamba_30k [--checkpoint FILE]
    python -m ecgr ec57      --model resumamba_30k [--dbs mitdb] [--bxb-only]
    python -m ecgr all       --model resumamba_30k   # ssl -> cpc -> train -> stepeval -> ec57
    python -m ecgr compare   resumamba_30k resumamba_1m   # side-by-side EC57 table

The two self-supervised stages come first and use no labels. Both depend only on the
architecture, so they are skipped when weights already exist - in this run, or in the run
ECGR_SSL_RUN / ECGR_CPC_RUN names.
"""
import argparse
import os
import sys


def _add_model_arg(p, required=True):
    from .models import list_models
    p.add_argument('--model', choices=list_models(), required=required,
                   help='which model of the family to use')


def _existing(path):
    return path if path and os.path.exists(path) else None


def build_parser():
    p = argparse.ArgumentParser(prog='ecgr', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='stage', required=True)

    sub.add_parser('config', help='print the resolved configuration')
    sub.add_parser('models', help='list the model family with parameter counts')

    d = sub.add_parser('data', help='build npy batches and/or tfrecords')
    d.add_argument('--step', choices=['npy', 'tfrecord', 'all'], default='all')
    d.add_argument('--db', nargs='*', default=None, help='datasets (default: all training ones)')
    d.add_argument('--limit', type=int, default=None, help='first N records per split (smoke test)')
    d.add_argument('--workers', type=int, default=None,
                   help=f'processes for the npy build (default: config.WORKERS)')
    d.add_argument('--no-audit', action='store_true', help='skip the post-build split audit')

    s = sub.add_parser('ssl', help='self-supervised pretraining of the backbone '
                                   '(masked span + masked lead reconstruction)')
    _add_model_arg(s)
    s.add_argument('--epochs', type=int, default=None)
    s.add_argument('--batch-size', type=int, default=None)
    s.add_argument('--lr', type=float, default=None)
    s.add_argument('--steps-per-epoch', type=int, default=None, help='0 = a full pass')
    s.add_argument('--val-steps', type=int, default=40, help='0 = the whole eval split')

    c = sub.add_parser('cpc', help='self-supervised pretraining of the context encoder '
                                   '(InfoNCE)')
    _add_model_arg(c)
    c.add_argument('--epochs', type=int, default=None)
    c.add_argument('--batch-size', type=int, default=None,
                   help='also the number of negatives: the pool is batch x windows')
    c.add_argument('--lr', type=float, default=1e-3)
    c.add_argument('--steps-per-epoch', type=int, default=None, help='0 = a full pass')
    c.add_argument('--val-steps', type=int, default=40, help='0 = the whole eval split')

    t = sub.add_parser('train', help='train one model')
    _add_model_arg(t)
    t.add_argument('--epochs', type=int, default=None)
    t.add_argument('--batch-size', type=int, default=None)
    t.add_argument('--lr', type=float, default=None)
    t.add_argument('--loss', choices=['wce', 'poly', 'poly2'], default=None,
                   help='default: config.LOSS (poly2, the paper\'s)')
    t.add_argument('--monitor', default=None,
                   help='default: config.MONITOR. val_loss is WRONG with poly2 - it bottoms '
                        'out at epoch 1 while the F1 keeps climbing')
    t.add_argument('--patience', type=int, default=None)
    t.add_argument('--cm-interval', type=int, default=1, help='epochs between F1 exports')
    t.add_argument('--ckpt-start-epoch', type=int, default=None,
                   help='first epoch (1-indexed) allowed to write a checkpoint; earlier '
                        'epochs are measured but not saved, and EarlyStopping is held off '
                        'until then so a checkpoint always exists (default: config value)')
    t.add_argument('--ssl-weights', default=None,
                   help='SSL backbone weights (default: this run\'s, else ECGR_SSL_RUN\'s, '
                        'else train the backbone from scratch)')
    t.add_argument('--ctx-weights', default=None,
                   help='CPC weights for the context encoder (default: this run\'s, else '
                        'ECGR_CPC_RUN\'s, else train from scratch)')
    t.add_argument('--ctx-trainable', action='store_true',
                   help='fine-tune the context encoder instead of freezing it (ablation)')
    t.add_argument('--freeze-backbone-epochs', type=int, default=None,
                   help='hold the SSL backbone frozen for this many epochs so the random '
                        'head cannot wash out the pretraining (default: config value)')

    e = sub.add_parser('stepeval', help='step-level metrics of a checkpoint on the eval split')
    _add_model_arg(e)
    e.add_argument('--checkpoint', default=None, help='default: best BEST_F1 of this run')
    e.add_argument('--batch-size', type=int, default=None)

    b = sub.add_parser('ec57', help='beat-level EC57 (bxb) over physionet + portal beat-eval')
    _add_model_arg(b)
    b.add_argument('--checkpoint', default=None, help='default: best BEST_F1 of this run')
    b.add_argument('--tag', default=None, help='report folder name (default: --model)')
    b.add_argument('--dbs', nargs='*', default=None)
    b.add_argument('--max-records', type=int, default=None)
    b.add_argument('--s-boost', type=float, default=1.0,
                   help='multiply the S probability before argmax; calibrate on portal only')
    b.add_argument('--lead-mode', choices=['auto', 'duplicate', 'native'], default=None,
                   help="how to fill the lead axis: 'duplicate' repeats the annotated lead "
                        "(the only option on the 2-lead EC57 databases), 'native' uses the "
                        "record's own montage, 'auto' picks per record (default: config)")
    b.add_argument('--bxb-only', action='store_true', help='re-score stored predictions')
    b.add_argument('--skip-physionet', action='store_true')
    b.add_argument('--skip-portal', action='store_true', help='skip the beat-eval holdout')
    b.add_argument('--splits', nargs='*', choices=['train', 'eval'], default=None,
                   help='portal splits to sample and score with bxb (default: config, '
                        'train and eval; `--splits` with nothing after it scores none)')
    b.add_argument('--split-records', type=int, default=None,
                   help='records sampled per split (default: config, 5000; 0 = all)')
    b.add_argument('--whole-record', action='store_true',
                   help='portal set: score the whole strip, not just the reviewed window')

    a = sub.add_parser('all', help='ssl -> cpc -> train -> stepeval -> ec57 for one model')
    _add_model_arg(a)
    a.add_argument('--epochs', type=int, default=None)
    a.add_argument('--batch-size', type=int, default=None)
    a.add_argument('--lr', type=float, default=None)
    a.add_argument('--patience', type=int, default=None)
    a.add_argument('--ckpt-start-epoch', type=int, default=None)
    a.add_argument('--skip-ssl', action='store_true',
                   help='reuse an existing backbone instead of pretraining one')
    a.add_argument('--skip-cpc', action='store_true',
                   help='reuse an existing context encoder instead of pretraining one')

    cmp_ = sub.add_parser('compare', help='side-by-side EC57 table of several tags')
    cmp_.add_argument('tags', nargs='+')
    cmp_.add_argument('--ec57-root', default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from . import config

    # Before ANY TensorFlow import: XLA reads XLA_FLAGS when it first compiles, and without
    # a libdevice path every training stage on a pip-CUDA install dies at its first step.
    if args.stage not in ('config', 'compare'):
        from . import xla
        xla.ensure_libdevice()

    if args.stage == 'config':
        print(config.describe())
        return 0

    if args.stage == 'models':
        from . import models
        for name in models.list_models():
            m = models.build(name)
            backbone = models.sub_model(m, 'backbone').count_params()
            ctx = models.sub_model(m, 'context_encoder').count_params()
            total = m.count_params()
            print(f"{name:16s} {m.name:26s} {total:>10,} params "
                  f"(backbone {backbone:>9,}, context {ctx:>7,}, head {total - backbone - ctx:>7,})"
                  f"  in={m.input_shape[1:]} out={m.output_shape[1:]}"
                  f"{'' if total < models.BUDGETS[name] else '  OVER BUDGET'}")
        return 0

    if args.stage == 'data':
        from .data import build_npy, build_tfrecord, splits
        if args.step in ('npy', 'all'):
            build_npy.build_all(args.db, limit=args.limit, workers=args.workers)
        if args.step in ('tfrecord', 'all'):
            build_tfrecord.build_all(args.db)
        if not args.no_audit and args.step in ('npy', 'all'):
            splits.audit_written_data(args.db)
        return 0

    if args.stage == 'compare':
        from .evaluation import report
        report.compare(args.tags, args.ec57_root)
        return 0

    config.ensure_run_dirs()

    if args.stage == 'ssl':
        from .training import ssl
        ssl.pretrain(args.model, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr,
                     steps_per_epoch=(config.SSL_STEPS_PER_EPOCH
                                      if args.steps_per_epoch is None
                                      else args.steps_per_epoch),
                     val_steps=args.val_steps)
        return 0

    if args.stage == 'cpc':
        from .training import cpc
        cpc.pretrain(args.model, epochs=args.epochs or config.CPC_EPOCHS,
                     batch_size=args.batch_size, lr=args.lr,
                     steps_per_epoch=(config.CPC_STEPS_PER_EPOCH
                                      if args.steps_per_epoch is None
                                      else args.steps_per_epoch),
                     val_steps=args.val_steps)
        return 0

    if args.stage == 'train':
        from .training import train as trainer
        trainer.train(args.model, epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, loss=args.loss, monitor=args.monitor,
                      patience=args.patience, cm_interval=args.cm_interval,
                      ckpt_start_epoch=args.ckpt_start_epoch,
                      ssl_weights=args.ssl_weights, ctx_weights=args.ctx_weights,
                      freeze_ctx=not args.ctx_trainable,
                      freeze_backbone_epochs=args.freeze_backbone_epochs)
        return 0

    if args.stage in ('stepeval', 'ec57'):
        from . import checkpoints, models
        ckpt = args.checkpoint or checkpoints.best_checkpoint(models.keras_name(args.model))
        print(f"checkpoint: {ckpt}")

        if args.stage == 'stepeval':
            from .training import train as trainer
            trainer.evaluate_checkpoint(ckpt, batch_size=args.batch_size)
        else:
            from .evaluation import ec57
            ec57.run(ckpt, tag=args.tag or args.model, dbs=args.dbs,
                     max_records=args.max_records, s_boost=args.s_boost,
                     bxb_only=args.bxb_only, skip_physionet=args.skip_physionet,
                     skip_portal=args.skip_portal, mark_window=not args.whole_record,
                     lead_mode=args.lead_mode, splits=args.splits,
                     split_records=args.split_records)
        return 0

    if args.stage == 'all':
        from . import checkpoints, models
        from .evaluation import ec57
        from .training import cpc, ssl, train as trainer
        keras_name = models.keras_name(args.model)

        ssl_weights = _existing(config.ssl_weights_dir(keras_name))
        if ssl_weights:
            print(f"{args.model}: backbone present ({ssl_weights}), skipping ssl")
        elif not args.skip_ssl:
            ssl_weights = ssl.pretrain(args.model, batch_size=args.batch_size)

        ctx_weights = _existing(config.cpc_weights_dir(keras_name))
        if ctx_weights:
            print(f"{args.model}: context encoder present ({ctx_weights}), skipping cpc")
        elif not args.skip_cpc:
            ctx_weights = cpc.pretrain(args.model, epochs=config.CPC_EPOCHS,
                                       batch_size=args.batch_size,
                                       steps_per_epoch=config.CPC_STEPS_PER_EPOCH)

        model = trainer.train(args.model, epochs=args.epochs, batch_size=args.batch_size,
                              lr=args.lr, patience=args.patience,
                              ckpt_start_epoch=args.ckpt_start_epoch,
                              ssl_weights=ssl_weights, ctx_weights=ctx_weights)
        ckpt = checkpoints.best_checkpoint(model.name)
        trainer.evaluate_checkpoint(ckpt, batch_size=args.batch_size)
        ec57.run(ckpt, tag=args.model)
        return 0

    return 1


if __name__ == '__main__':
    sys.exit(main())
