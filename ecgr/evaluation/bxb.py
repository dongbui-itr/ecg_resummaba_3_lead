"""Driving the WFDB `bxb` / `sumstats` tools, which are what EC57 actually is.

Only the beat comparison is wired up - this project classifies beats and nothing else - so
the SV-event, IVCD, AFib and rhythm branches of the original driver are gone rather than
carried along unused.

Three things differ from the code this replaces:
  * the shell scripts are located from the package (config.WFDB_SCRIPTS_DIR), not from
    os.getcwd(); the old version silently scored nothing when run from another directory
  * the scoring happens in a caller-supplied working directory, never in the Physionet
    folders themselves, so two models cannot overwrite each other's annotations
  * every argument is shell-quoted and the exit status is checked. The paths here contain a
    run tag and a dataset name, and the portal dataset names contain spaces ("dataset 2_3_4
    - AFib - v2"), so an unquoted `shell=True` command line splits a directory in half and
    the script scores an empty folder - which looks exactly like a model that found no beats.
"""
import os
import shlex
import shutil
import subprocess

from .. import config

# bxb's default 5-minute learning period is what EC57 prescribes for the long Physionet
# records. On a strip shorter than that the default start time leaves an empty interval and
# bxb exits with 'improper interval specified', so short records need the -f 0 variants.
SCRIPT_FULL = 'bxb-script.sh'             # Physionet records (30 min+)
SCRIPT_SHORT = 'bxb-script2.sh'           # short strips, -f 0
SCRIPT_MARK_WINDOW = 'bxb-script-mark-window.sh'   # short strips, reviewed window only


def have_wfdb_tools():
    """True when bxb and sumstats are on PATH - checked up front, because the shell scripts
    fail silently (they write no report) when they are not."""
    return all(shutil.which(tool) for tool in ('bxb', 'sumstats'))


def run_bxb(db_name, work_dir, report_root, ref_ext, ai_ext, script=SCRIPT_FULL):
    """Score `work_dir` with bxb + sumstats; reports land in <report_root>/<db_name>/.

    work_dir must hold the records, their reference annotations and the .<ai_ext>
    predictions, flat, one set per record.
    """
    script_path = os.path.join(config.WFDB_SCRIPTS_DIR, script)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"bxb driver not found: {script_path}")
    if not have_wfdb_tools():
        raise RuntimeError("bxb / sumstats are not on PATH - install the WFDB applications")

    out_dir = os.path.join(report_root, db_name)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(report_root, exist_ok=True)

    command = ' '.join(shlex.quote(part) for part in (
        script_path, work_dir + '/', out_dir + '/', ref_ext, ai_ext,
        f"{db_name}_QRS_report_line", f"{db_name}_QRS_report_standard"))
    status = subprocess.call(command, shell=True)
    if status != 0:
        print(f"  {os.path.basename(script_path)} exited with status {status}")

    # The scripts leave their scratch .out files behind in the scoring directory
    for name in os.listdir(work_dir):
        if name.endswith('.out') or name.startswith('sd.'):
            os.remove(os.path.join(work_dir, name))

    report = os.path.join(out_dir, f"{db_name}_QRS_report_line.out")
    if not os.path.exists(report):
        print(f"  no report written to {report} - bxb produced nothing for {db_name}")
        return None
    return report
