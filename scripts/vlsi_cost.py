#!/usr/bin/env python
"""What the Mamba -> CfC swap costs in silicon, per model.

    python scripts/vlsi_cost.py [--sizes 2m 1m 100k 30k]

Only the TIME MIXER is compared, because it is the only thing that changes: the stem, the
ResU path, the convolutions inside each block, the AdaIN conditioning and the head are
identical in both families, so counting them would dilute the number the swap is about.

What is counted, per output step of a 10 s strip (500 steps at 20 ms):

    mac_per_step   multiply-accumulates the mixer issues per step, all channels
    coeff_words    coefficients that have to sit in ROM
    state_words    runtime memory: a FIR's delay line, or a CfC's state registers
    params         trained weights in the .keras file (NOT what reaches silicon for the SSM,
                   whose 4 scalars per oscillator are baked into coeff_words before export)

The honest caveat is printed with the table: a CfC step needs transcendental functions
(softplus, exp, two tanh) that an FIR does not, so it trades multipliers and SRAM for
LUT/CORDIC area. The trade is only worth it because the SSM side is memory-bound, not
arithmetic-bound.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mixers(model):
    from ecgr.models.layers import CfC1D, DiagSSM1D
    found = []

    def walk(layer):
        if isinstance(layer, (CfC1D, DiagSSM1D)):
            found.append(layer)
        for sub in getattr(layer, 'layers', []):
            walk(sub)
    walk(model)
    return found


def totals(model):
    acc = {'layers': 0, 'mac_per_step': 0, 'coeff_words': 0, 'state_words': 0, 'params': 0,
           'nonlin_per_step': 0}
    for m in mixers(model):
        c = m.cost()
        acc['layers'] += 1
        for k in ('mac_per_step', 'coeff_words', 'state_words', 'params'):
            acc[k] += c.get(k, 0)
        acc['nonlin_per_step'] += sum(c.get('nonlin_per_step', {}).values())
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', nargs='*', default=['2m', '1m', '100k', '30k'])
    ap.add_argument('--csv', default=None, help='also write the table here')
    args = ap.parse_args()

    from ecgr import xla
    xla.ensure_libdevice()
    from ecgr import models

    rows = []
    for size in args.sizes:
        ssm = models.build(f'resumamba_{size}')
        cfc = models.build(f'resucfc_{size}')
        a, b = totals(ssm), totals(cfc)
        rows.append((size, a, b, ssm.count_params(), cfc.count_params()))

    hdr = (f"{'size':6s} {'mixer':5s} {'blocks':>6s} {'MAC/step':>10s} {'coeff ROM':>10s} "
           f"{'state':>8s} {'mixer par':>9s} {'model par':>10s} {'nonlin/step':>11s}")
    print(hdr)
    print('-' * len(hdr))
    for size, a, b, pa, pb in rows:
        print(f"{size:6s} {'ssm':5s} {a['layers']:6d} {a['mac_per_step']:10,} "
              f"{a['coeff_words']:10,} {a['state_words']:8,} {a['params']:9,} {pa:10,} "
              f"{a['nonlin_per_step']:11,}")
        print(f"{size:6s} {'cfc':5s} {b['layers']:6d} {b['mac_per_step']:10,} "
              f"{b['coeff_words']:10,} {b['state_words']:8,} {b['params']:9,} {pb:10,} "
              f"{b['nonlin_per_step']:11,}")
        f = lambda k: (a[k] / b[k]) if b[k] else float('inf')
        print(f"{size:6s} {'x':5s} {'':6s} {f('mac_per_step'):9.1f}x "
              f"{f('coeff_words'):9.1f}x {f('state_words'):7.1f}x "
              f"{f('params'):8.1f}x {pa / pb:9.2f}x")
        print()

    print("Read it with the caveat: a CfC step issues the transcendentals an FIR does not")
    print("(softplus + exp + 2 tanh per cell, the right-hand column), so this is multipliers")
    print("and SRAM traded for LUT or CORDIC area. It is worth it because the SSM side is")
    print("memory-bound - the delay line and the coefficient ROM, not the MACs, are what")
    print("decide whether the block fits.")

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['size', 'mixer', 'mixer_layers', 'mac_per_step', 'coeff_words',
                        'state_words', 'mixer_params', 'model_params', 'nonlin_per_step'])
            for size, a, b, pa, pb in rows:
                for tag, c, p in (('ssm', a, pa), ('cfc', b, pb)):
                    w.writerow([size, tag, c['layers'], c['mac_per_step'], c['coeff_words'],
                                c['state_words'], c['params'], p, c['nonlin_per_step']])
        print(f"\n-> {args.csv}")


if __name__ == '__main__':
    main()
