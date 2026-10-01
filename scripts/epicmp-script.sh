#! /bin/sh
# epicmp over a rhythm scoring folder - long / native EC57 records (default 5 min start,
# same default epicmp itself applies, matching bxb-script.sh).
#
# Same argument order as bxb-script.sh, minus the second (matrix-format) report: epicmp's
# -A report is already the one sumstats reads for the Gross episode/duration Se/+P line.
#
#   $1 DB_PATH    scoring directory, one record's .hea/.dat/.<EXT_REF>/.<EXT_AI> per name
#   $2 REPORT_PATH  where the aggregated report is moved to
#   $3 EXT_REF    reference annotation extension (e.g. atr, rhy)
#   $4 EXT_AI     hypothesis annotation extension (e.g. rhi)
#   $5 OUTPUT_NAME  base name of the aggregated report (no extension)
#   $6 EXTRA      optional extra epicmp flags (unquoted), e.g. -x

DB_PATH=$1
REPORT_PATH=$2
EXT_REF=$3
EXT_AI=$4
OUTPUT_NAME=$5
EXTRA=$6          # optional epicmp flags, e.g. -x (reference AFL out of AFIB +P)

OUTPUT="epi"
cd "$DB_PATH" || exit 1
echo "$DB_PATH"

for entry in *."$EXT_AI" ; do
      name=$(echo "$entry" | cut -f 1 -d '.')
      echo "epicmp $name"
      epicmp -r "$name" -a "$EXT_REF" "$EXT_AI" $EXTRA -L -A "$OUTPUT".out
    done
sumstats "$OUTPUT".out >> "$OUTPUT_NAME".out

if [ ! -d "$REPORT_PATH" ]; then
    mkdir "$REPORT_PATH"
fi
mv "$OUTPUT_NAME".out "$REPORT_PATH"
