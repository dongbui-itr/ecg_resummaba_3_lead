"""ecgr.rhythm - per-second rhythm classification on 3-lead strips, ResUMamba backbone.

Contract: in (2500, 3) = 10 s at 250 Hz over three leads; two outputs:
    'rhythm' (10, 5) - per second, SINUS / AFIB / SVT / VT / AVB (2nd or 3rd degree), softmax
    'lead'   (4,)    - per window, NOISE (no readable lead) / CH1 / CH2 / CH3 = the lead with
                       the best signal, softmax

    records --> data (train / eval / test windows) --> train --> eval --> predict

The labels follow the reference rhythm project (itr-ai-sensor_annotation-ae_ecg_classification:
same six classes, same caliper-span convention) cut to one label per second, and the model is
the beat family's backbone with a one-second head (model.py). Noise and lead order are
randomised per batch (augment.py), and the noise that was added defines the lead target.

Holdout: every study of the rhythm test folder (config.EVAL_DIR) and of the v4 eval list is
removed before the train/eval split; train and eval are split by study hash. Both are proven
before writing (inventory.verify_split) and again from the written files (build.audit_written).

    python -m ecgr.rhythm --help
"""
