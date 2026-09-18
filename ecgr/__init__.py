"""ecgr - 3-lead ECG beat detection + AAMI classification with ResUMamba.

One model family (the ResUMamba paper adapted to a seq2seq contract) in four sizes - 2M, 1M,
100K and 30K parameters - and one path through the data:

    records --> npy --> tfrecord --> ssl --> cpc --> train --> step eval --> beat eval (EC57)

Contract: in (2500, 3) = 10 s at 250 Hz over three leads, out (500, 4) softmax = detection
AND AAMI classification at 20 ms resolution, with no R peaks given.

The `ssl` and `cpc` stages are what distinguish this family: both are self-supervised on
unlabeled signal. `ssl` pretrains the backbone by masked-span and masked-lead reconstruction;
`cpc` trains the context encoder by InfoNCE and it is then frozen, so the beat loss never
reshapes it.

Every stage is reachable from one command line, `python -m ecgr <stage>`; see README.md.
"""
__version__ = "2.0.0"
