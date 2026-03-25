#!/bin/bash

# Check if both files are provided as arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file1.csv> <file2.csv>"
    exit 1
fi

FILE1=$1
FILE2=$2

echo "Comparing $FILE1 and $FILE2..."

# We use 'paste' to read both files side-by-side.
# 'tr -d "\r"' ensures that Windows carriage returns are removed so they don't cause false mismatches.
# 'awk' is then used to compare the columns row by row.
paste <(tr -d '\r' < "$FILE1") <(tr -d '\r' < "$FILE2") | awk '
BEGIN { 
    matches = 0; 
    total = 0; 
}
{
    total++
    # If column 1 equals column 2, it is a match
    if ($1 == $2) {
        matches++
    }
}
END {
    if (total == 0) {
        print "Files are empty."
        exit
    }
    printf "Total rows compared: %d\n", total
    printf "Exact matches:       %d\n", matches
    printf "Mismatches:          %d\n", total - matches
    printf "Agreement Rate:      %.2f%%\n", (matches/total)*100
}'
