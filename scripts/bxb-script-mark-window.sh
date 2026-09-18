#! /bin/sh
# bxb over a portal eval folder, comparing ONLY inside each record's reviewed window.
#
# Every portal strip carries the window the reviewer actually worked on as two .hea
# comments:
#
#     # startMarkSample: 3564
#     # stopMarkSample: 8563
#
# Both are sample numbers at the record's own sampling rate, so they are handed to bxb as
# WFDB sample-time strings (-f sSTART -t sSTOP). Scoring the whole record instead would
# pool in beats outside the reviewed region and dilute the statistics - on a typical strip
# the window holds every S beat but only a fraction of the N beats.
#
# Same arguments and output files as bxb-script.sh / bxb-script2.sh, so it is a drop-in
# choice for EC57ShellScripts.ec57_eval(bxb_script=...).

DB_PATH=$1
REPORT_PATH=$2
EXT_REF=$3
EXT_AI=$4
OUTPUT_NAME=$5
OUTPUT_NAME2=$6

OUTPUT="bxb"
cd "$DB_PATH" || exit 1
echo "$DB_PATH"

scored=0
skipped=0
for entry in *."$EXT_AI" ; do
      name=$(echo "$entry" | cut -f 1 -d '.')
      header="$name.hea"

      if [ ! -f "$header" ]; then
          echo "skip $name: no $header" >&2
          skipped=$((skipped + 1))
          continue
      fi

      # tr -dc keeps just the digits of '# startMarkSample: 3564'
      START=$(grep -m1 -i 'startMarkSample' "$header" | tr -dc '0-9')
      STOP=$(grep -m1 -i 'stopMarkSample' "$header" | tr -dc '0-9')

      if [ -z "$START" ] || [ -z "$STOP" ] || [ "$STOP" -le "$START" ]; then
          echo "skip $name: unusable mark window (start='$START' stop='$STOP')" >&2
          skipped=$((skipped + 1))
          continue
      fi

      bxb -r "$name" -a "$EXT_REF" "$EXT_AI" -f s"$START" -t s"$STOP" -L "$OUTPUT".out sd.out
      bxb -r "$name" -a "$EXT_REF" "$EXT_AI" -f s"$START" -t s"$STOP" -S "$OUTPUT_NAME2".out
      scored=$((scored + 1))
    done

echo "bxb mark-window: scored $scored records, skipped $skipped"
sumstats "$OUTPUT".out >> "$OUTPUT_NAME".out

if [ ! -d "$REPORT_PATH" ]; then
    mkdir "$REPORT_PATH"
fi
mv "$OUTPUT_NAME".out "$REPORT_PATH"
mv "$OUTPUT_NAME2".out "$REPORT_PATH"
