#!/bin/bash

LOGDIR="logs"
OUTPUT="$LOGDIR/all_logs_combined.log"

echo "######### Combined HHT Logs #########" > "$OUTPUT"
echo "" >> "$OUTPUT"

for file in "$LOGDIR"/*.log; do
    # Skip the output file itself if it already exists
    if [ "$(basename "$file")" = "$(basename "$OUTPUT")" ]; then
        continue
    fi

    echo "------ START $(basename "$file") ------" >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
    echo "------ END $(basename "$file") ------" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

echo "Wrote combined logs to: $OUTPUT"
