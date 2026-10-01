"""Driving the WFDB `epicmp` / `sumstats` tools for RHYTHM EPISODE comparison.

This is the rhythm task's counterpart of bxb.py: bxb compares individual beats, epicmp
compares rhythm EPISODES (onset/offset of AFIB, SVT, VT, ... against a reference) and reports
episode sensitivity/+P and duration sensitivity/+P (the "AFib" style report in EC57 - the
same tool scores any rhythm class whose aux_note matches between reference and hypothesis,
not only AFIB).

Same three departures from the shell scripts this was adapted from (see bxb.py's docstring
for why):
  * the scripts are located from config.WFDB_SCRIPTS_DIR, not os.getcwd()
  * scoring happens in a caller-supplied working directory, never in the source folders
  * every argument is shell-quoted and the exit status is checked
"""
import os
import shlex
import shutil
import subprocess

from .. import config

# epicmp's default 5-minute learning period matches bxb's; the same "too short, empty
# interval" failure applies to strips shorter than that, so short records need -f 0 too.
SCRIPT_FULL = 'epicmp-script.sh'      # Physionet records (30 min+)
SCRIPT_SHORT = 'epicmp-script2.sh'    # short strips / 10 s windows, -f 0


def have_wfdb_tools():
    """True when epicmp and sumstats are on PATH - the shell scripts fail silently (they
    write no report) when they are not."""
    return all(shutil.which(tool) for tool in ('epicmp', 'sumstats'))


def run_epicmp(db_name, work_dir, report_root, ref_ext, ai_ext, script=SCRIPT_FULL,
              label='rhythm', quiet=False, extra_flags=()):
    """Score `work_dir` with epicmp + sumstats; reports land in <report_root>/<db_name>/.

    work_dir must hold the records, their reference rhythm annotations (.<ref_ext>) and the
    .<ai_ext> predictions, flat, one set per record. `label` names the report
    (<db_name>_<label>_report_line.out) - the rhythm class under test. `quiet` drops the
    script's per-record echo. `extra_flags` go to every epicmp call (e.g. ('-x',)).
    Returns the report path, or None if nothing was scored.
    """
    script_path = os.path.join(config.WFDB_SCRIPTS_DIR, script)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"epicmp driver not found: {script_path}")
    if not have_wfdb_tools():
        raise RuntimeError("epicmp / sumstats are not on PATH - install the WFDB applications")

    # Only this label's report is replaced: one database is scored once per rhythm class,
    # each call adding its own <db>_<class>_report_line.out next to the others.
    out_dir = os.path.join(report_root, db_name)
    os.makedirs(out_dir, exist_ok=True)
    report_name = f"{db_name}_{label}_report_line"
    for stale in (os.path.join(out_dir, report_name + '.out'),
                  os.path.join(work_dir, report_name + '.out')):
        if os.path.exists(stale):
            os.remove(stale)
    command = ' '.join(shlex.quote(part) for part in (
        script_path, work_dir + '/', out_dir + '/', ref_ext, ai_ext, report_name,
        ' '.join(extra_flags)))
    # quiet also drops stderr: -f 0 makes epicmp print "nonstandard comparison selected" once
    # per record, thousands of times over the rhythm_eval strips. The exit status still tells.
    sink = subprocess.DEVNULL if quiet else None
    status = subprocess.call(command, shell=True, stdout=sink, stderr=sink)
    if status != 0:
        print(f"  {os.path.basename(script_path)} exited with status {status}")

    # The scripts leave their scratch .out files behind in the scoring directory
    for name in os.listdir(work_dir):
        if name.endswith('.out'):
            os.remove(os.path.join(work_dir, name))

    report = os.path.join(out_dir, f"{report_name}.out")
    if not os.path.exists(report):
        print(f"  no report written to {report} - epicmp produced nothing for {db_name}")
        return None
    return report
