#!/bin/bash
# Auto-processing script that runs after Fusion 360 dataset download completes

set -e

PROJECT_DIR="/mnt/d/AI/HermesAgent/Hermes_WorkSpace/CTO/mesh-topo-ai"
DATA_DIR="$PROJECT_DIR/data/fusion360"
ZIP_FILE="$DATA_DIR/fusion360_segmentation.zip"
EXTRACT_DIR="$DATA_DIR/extracted"
OUTPUT_DIR="$DATA_DIR/processed"
REPORT_FILE="$DATA_DIR/preprocessing_report.json"

# Expected file size (3.1GB = 3326258457 bytes)
EXPECTED_SIZE=3326258457

echo "[$(date)] Checking download status..."

if [ ! -f "$ZIP_FILE" ]; then
    echo "[$(date)] ZIP file not found. Exiting."
    exit 1
fi

ACTUAL_SIZE=$(stat -c%s "$ZIP_FILE")

if [ "$ACTUAL_SIZE" -lt "$EXPECTED_SIZE" ]; then
    echo "[$(date)] Download incomplete ($ACTUAL_SIZE / $EXPECTED_SIZE bytes). Exiting."
    exit 0
fi

echo "[$(date)] Download complete! Starting auto-processing..."

# Step 1: Extract
echo "[$(date)] Step 1: Extracting dataset..."
mkdir -p "$EXTRACT_DIR"
cd "$EXTRACT_DIR"
unzip -q "$ZIP_FILE"
echo "[$(date)] Extraction complete."

# Step 2: Inspect structure
echo "[$(date)] Step 2: Inspecting dataset structure..."
find "$EXTRACT_DIR" -name "*.obj" -o -name "*.ply" -o -name "*.stl" | head -20 > "$DATA_DIR/sample_files.txt"
OBJ_COUNT=$(find "$EXTRACT_DIR" -name "*.obj" | wc -l)
echo "[$(date)] Found $OBJ_COUNT .obj files."

# Step 3: Run batch preprocessing (density analysis only)
echo "[$(date)] Step 3: Running density analysis..."
cd "$PROJECT_DIR"
python3 -c "
import sys
sys.path.insert(0, '.')
from tokenizer.scripts.batch_preprocess import preprocess_batch
from pathlib import Path

stats = preprocess_batch(
    input_dir=Path('$EXTRACT_DIR'),
    output_dir=Path('$OUTPUT_DIR'),
    min_patch_density=20.0,
    max_patches_per_mesh=50,
)

import json
with open('$REPORT_FILE', 'w') as f:
    json.dump(stats, f, indent=2, default=str)

print('\\n' + '='*60)
print('DENSITY ANALYSIS COMPLETE')
print('='*60)
print(f'Total files: {stats[\"total\"]}')
print(f'Processed: {stats[\"processed\"]}')
print(f'Discarded (UV density): {stats[\"discarded_by_uv_density\"]}')
print(f'Discarded (too many patches): {stats[\"discarded_by_too_many_patches\"]}')
if stats['processed'] > 0:
    discard_rate = (stats['discarded_by_uv_density'] + stats['discarded_by_too_many_patches']) / max(stats['total'], 1)
    print(f'Overall discard rate: {discard_rate*100:.1f}%')
print('='*60)
"

echo "[$(date)] Auto-processing complete!"
echo "[$(date)] Report saved to: $REPORT_FILE"
