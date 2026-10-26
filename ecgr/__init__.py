"""ecgr - 3-lead ECG beat, rhythm, channel and noise model (rhythm_u3cfc) with EC57 / v4 scoring.

Model (ecgr/rhythm/u3cfc.py): in (2500, 3) = 10 s at 250 Hz over three leads.
  encoder  three U-Nets - QRS morphology, P / T slow waves, atrial (F / f) on the QRST-cancelled
           residual - plus the template-residual quality block; out (1250, x). <= 900K params.
  decoder  CNN + CfC only: beat (1250, 4), rhythm (20, 5), channel (5, 4), noise (5, 2).
           <= 300K params.

    python -m ecgr.rhythm data | audit                 # build / re-check the training windows
    python -m ecgr.rhythm train --model rhythm_u3cfc_1m ...           (scripts/run_u3cfc.sh)
    python -m ecgr.rhythm ec57 --checkpoint final_model.keras --tag T # rhythm epicmp
    python scripts/dualcfc_bxb.py --checkpoint final_model.keras --tag T [--noise-gate 0.2]
                                                       # beat bxb: EC57 databases + v4
    python scripts/u3cfc_noise_report.py --checkpoint final_model.keras   # nstdb vs true SNR

Rhythm EC57 is decoded with ECGR_RHYTHM_DECODE_NOISE_THR=1.01: masking seconds by the noise head
removes VT episodes. ecgr/evaluation is the shared bxb / epicmp scoring.
"""
__version__ = "3.0.0"
