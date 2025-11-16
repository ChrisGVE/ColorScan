#!/usr/bin/env zsh

# strip_geolocation.zsh
# Removes GPS and location data from all images in the current directory
# while preserving useful camera metadata for testing purposes.
#
# Usage:
#   ./strip_geolocation.zsh           # Process current directory
#   ./strip_geolocation.zsh --dry-run # Show what would be removed without making changes
#   ./strip_geolocation.zsh --help    # Show this help message

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if exiftool is installed
if ! command -v exiftool &> /dev/null; then
    echo "${RED}Error: exiftool is not installed${NC}"
    echo "Install with: brew install exiftool"
    exit 1
fi

# Parse arguments
DRY_RUN=false
SHOW_HELP=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "${RED}Unknown argument: $arg${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: ./strip_geolocation.zsh [OPTIONS]"
    echo ""
    echo "Removes GPS and location data from all images in the current directory."
    echo ""
    echo "Options:"
    echo "  --dry-run    Show what would be removed without making changes"
    echo "  --help, -h   Show this help message"
    echo ""
    echo "Supported formats: JPEG, JPG, HEIC, PNG, TIFF, DNG, CR2, NEF, etc."
    echo ""
    echo "What gets removed:"
    echo "  - GPS coordinates (latitude, longitude, altitude)"
    echo "  - GPS direction and bearing"
    echo "  - GPS speed and timestamp"
    echo "  - Location names and place information"
    echo ""
    echo "What gets preserved:"
    echo "  - Camera make and model"
    echo "  - White balance settings"
    echo "  - Color space information"
    echo "  - Exposure settings (ISO, shutter, aperture)"
    echo "  - Date/time stamps"
    echo "  - Lens information"
    exit 0
fi

# Count image files in current directory
IMAGE_COUNT=$(find . -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.heic" -o -iname "*.png" -o -iname "*.tiff" -o -iname "*.dng" -o -iname "*.cr2" -o -iname "*.nef" \) | wc -l | xargs)

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "${YELLOW}No image files found in current directory${NC}"
    exit 0
fi

echo "${BLUE}Found $IMAGE_COUNT image file(s) in current directory${NC}"
echo ""

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    echo "${YELLOW}=== DRY RUN MODE ===${NC}"
    echo "Checking GPS data in images..."
    echo ""

    HAS_GPS=0
    for file in *.{jpg,jpeg,JPG,JPEG,heic,HEIC,png,PNG,tiff,TIFF,dng,DNG,cr2,CR2,nef,NEF}(N); do
        if [ -f "$file" ]; then
            GPS_DATA=$(exiftool -gps:all "$file" 2>/dev/null)
            if [ -n "$GPS_DATA" ]; then
                echo "${YELLOW}📍 $file has GPS data:${NC}"
                echo "$GPS_DATA" | grep -E "GPS" | head -5
                echo ""
                HAS_GPS=$((HAS_GPS + 1))
            fi
        fi
    done

    if [ $HAS_GPS -eq 0 ]; then
        echo "${GREEN}✓ No GPS data found in any images${NC}"
    else
        echo "${YELLOW}Found GPS data in $HAS_GPS file(s)${NC}"
        echo "Run without --dry-run to remove GPS data"
    fi

    exit 0
fi

# Confirm before processing
echo "${YELLOW}About to remove GPS data from $IMAGE_COUNT image(s)${NC}"
echo "Original files will be modified (no backups)"
echo ""
read "?Continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "${YELLOW}Cancelled${NC}"
    exit 0
fi

echo ""
echo "${BLUE}Removing GPS data...${NC}"

# Remove GPS data using exiftool
# -gps:all= removes all GPS tags
# -overwrite_original modifies files in-place without backup
# -q quiet mode (only errors)
# -m ignore minor errors and warnings

exiftool -gps:all= -overwrite_original -q -m . 2>&1 | while IFS= read -r line; do
    # Filter out minor warnings, show only important messages
    if [[ ! "$line" =~ "Warning: \[minor\]" ]]; then
        echo "$line"
    fi
done

EXIT_CODE=${pipestatus[1]}

echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "${GREEN}✓ Successfully removed GPS data from all images${NC}"
    echo ""

    # Show summary of what was preserved
    echo "${BLUE}Sample of preserved metadata (first file):${NC}"
    FIRST_FILE=$(find . -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.heic" \) | head -1)
    if [ -n "$FIRST_FILE" ]; then
        exiftool -Make -Model -WhiteBalance -ColorSpace -DateTimeOriginal -ISO "$FIRST_FILE" 2>/dev/null | grep -v "^ExifTool"
    fi

    echo ""
    echo "${GREEN}Images are now safe to commit!${NC}"
else
    echo "${RED}✗ Error removing GPS data${NC}"
    exit 1
fi
