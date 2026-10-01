#! /bin/sh
# epicmp over a rhythm scoring folder - short strips / 10 s windows.
#
# epicmp's default comparison start is 5 minutes into the record, which leaves an empty
# interval on anything shorter and epicmp exits with "improper interval specified" - the
# same failure bxb has on short strips (see bxb-script2.sh). -f 0 starts at the beginning.
#
# Same argument order as epicmp-script.sh.

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
      epicmp -r "$name" -a "$EXT_REF" "$EXT_AI" -f 0 $EXTRA -L -A "$OUTPUT".out
    done
sumstats "$OUTPUT".out >> "$OUTPUT_NAME".out

if [ ! -d "$REPORT_PATH" ]; then
    mkdir "$REPORT_PATH"
fi
mv "$OUTPUT_NAME".out "$REPORT_PATH"
