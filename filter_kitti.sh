#!/bin/bash

emptyLine="DontCare 0 0 0 0 0 0 0 0 0 0 0 0 0 0"

# Loop through .txt files in the current directory
for file in *.txt; do
    # Check if the file is a regular file
    if [ -f "$file" ]; then
        # Extract the line starting with 'car' from the file
        car_line=$(grep -i '^car\s' "$file" | head -n 1)
        # echo "$file" "$car_line"
        # Create a temporary file to store the selected line
        #tmp_file=$(mktemp)
        #echo "$car_line" - "$file"
        # If 'car' line found, write it to the temporary file
        if [ -n "$car_line" ]; then
            echo "$file"
            echo "$car_line" > "$file"
        else
            echo "$file" - Empty
            echo "$emptyLine" > "$file"
        fi
        
        # Replace the original file with the temporary file
        # mv "$tmp_file" "$file"
    fi
done
