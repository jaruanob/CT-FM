#!/bin/bash

set -euo pipefail
# Input CSV
CSV_FILE='/home/vlab/Documents/Repositories/Foundational_models/CT-FM/database_dental.csv'
# Base output directory
OUTPUT_BASE="/home/vlab/Documents/Collections/Dental/CT-FM_data/"
PREVIEW_LINES=5


# check plastimatch exists
if ! command -v plastimatch >/dev/null 2>&1; then
  echo "ERROR: plastimatch not found in PATH. Install plastimatch or add it to PATH."
  exit 1
fi

mkdir -p "$OUTPUT_BASE"

# Determine the index of id_name and full_path from header (works even if columns move)
# We use awk to build mapping from header name -> column index
read -r header_line < "$CSV_FILE"
# sanitize header (remove \r if present)
header_line="${header_line//$'\r'}"
# Use awk to get indices
id_idx=$(awk -F',' 'NR==1{ for(i=1;i<=NF;i++){ gsub(/^[ \t]+|[ \t]+$/, "", $i); if($i=="id_name"){print i; exit} } }' "$CSV_FILE")
path_idx=$(awk -F',' 'NR==1{ for(i=1;i<=NF;i++){ gsub(/^[ \t]+|[ \t]+$/, "", $i); if($i=="full_path"){print i; exit} } }' "$CSV_FILE")

if [ -z "$id_idx" ] || [ -z "$path_idx" ]; then
  echo "ERROR: Could not find 'id_name' or 'full_path' column in header."
  echo "Header was: $header_line"
  exit 1
fi

echo "Detected columns: id_name -> column $id_idx, full_path -> column $path_idx"
echo

# Preview first PREVIEW_LINES mappings so you can confirm before conversion
echo "Preview (first $PREVIEW_LINES data rows):"
awk -F',' -v idcol="$id_idx" -v pcol="$path_idx" 'NR>1 && NR<=('"$PREVIEW_LINES"'+1){
    # join fields that may have been split if there are stray commas is not handled here,
    # but this CSV seems well-formed (full_path is a single field).
    id=$idcol; path=$pcol;
    gsub(/^"|"$/, "", id); gsub(/^"|"$/, "", path);
    gsub(/^[ \t]+|[ \t]+$/, "", id); gsub(/^[ \t]+|[ \t]+$/, "", path);
    print NR-1 ": id_name=\"" id "\"  full_path=\"" path "\"" 
}' "$CSV_FILE"
echo
read -r -p "Continue with conversion? (y/N) " ans
ans=${ans:-N}
if [[ ! "$ans" =~ ^[Yy]$ ]]; then
  echo "Aborting."
  exit 0
fi

# Process all rows
awk -F',' -v idcol="$id_idx" -v pcol="$path_idx" 'NR>1{
    id=$idcol; path=$pcol;
    # remove surrounding quotes
    gsub(/^"|"$/, "", id); gsub(/^"|"$/, "", path);
    # trim leading/trailing whitespace
    sub(/^[ \t\r\n]+/, "", id); sub(/[ \t\r\n]+$/, "", id);
    sub(/^[ \t\r\n]+/, "", path); sub(/[ \t\r\n]+$/, "", path);
    # print id and path using a delimiter unlikely to appear in names
    print id "\t" path
}' "$CSV_FILE" | while IFS=$'\t' read -r id_name full_path; do

    # skip empty
    if [ -z "$id_name" ] || [ -z "$full_path" ]; then
        echo "Skipping empty line (id_name='$id_name', full_path='$full_path')"
        continue
    fi

    OUTPUT_DIR="${OUTPUT_BASE%/}/${id_name}"   # ensure no double slash
    # mkdir -p "$OUTPUT_DIR"

    # Determine output filename
    base_name=$(basename "$full_path")
    # if full_path ends with a slash (directory), choose a default name
    if [[ "$full_path" == */ ]]; then
        outname="${id_name}.nrrd"
        OUTPUT_FILE="${OUTPUT_DIR}/${outname}"
        echo "Do nothing Output Dir: $OUTPUT_DIR"
        # # plastimatch expects --input <SORTED_DIR> when DICOM dir - we will pass directory directly
        # echo "Converting DICOM directory: $full_path -> $OUTPUT_FILE"
        # plastimatch convert-img --input "$full_path" --output-img "$OUTPUT_FILE" --format nrrd
    else
        OUTPUT_FILE="${OUTPUT_DIR}.nrrd"
        echo "OUTPUT_FILE: $OUTPUT_FILE"
        echo "Converting image file: $full_path -> $OUTPUT_FILE"
        # Use --input-img for nifti files (handles single image better)
        plastimatch convert --input "$full_path" --output-img "$OUTPUT_FILE" 
    fi

    # # spacing_line=$(plastimatch info "$OUTPUT_FILE" | grep -i "Spacing" | head -n 1 | awk -F'=' '{print $2}' | xargs)
    # spacing_line=$(plastimatch header "$OUTPUT_FILE" 2>/dev/null | grep -E "^[[:space:]]*Spacing" | head -n 1 | sed 's/.*Spacing[[:space:]]*=[[:space:]]*//' | xargs || true)

    # # Handle missing or malformed output
    # if [ -z "$spacing_line" ]; then
    #     echo "⚠️  Could not read spacing for $id_name — skipping spacing fix."
    #     spacing_line="unknown"
    # fi

    # echo "Detected spacing: $spacing_line"

    # if [[ "$spacing_line" == "1.0000 1.0000 1.0000" ]]; then
    #     DESIRED_SPACING="0.25 0.25 0.25"
    #     echo "⚙️  Fixing spacing to $DESIRED_SPACING ..."
    #     # plastimatch adjust --input "$OUTPUT_FILE" --output "$OUTPUT_FILE" --spacing $DESIRED_SPACING
    #     teem-unu axinfo -i "$OUTPUT_FILE" -a 0 -sp 0.25 \
    #         | teem-unu axinfo -a 1 -sp 0.25 \
    #         | teem-unu axinfo -a 2 -sp 0.25 -o "$OUTPUT_FILE"
    #     echo "✅ Spacing updated for $id_name"
    #     plastimatch header "$OUTPUT_FILE"
    # else
    #     echo "ℹ️  Spacing OK, no modification needed."
    # fi


    if [ $? -eq 0 ]; then
        echo "✅ OK: $id_name"
    else
        echo "❌ FAILED: $id_name"
    fi

    echo "--------------------------------------"

done

echo "All done."
