#!/bin/bash

# List of question indices
indices=(0 1 2 3 4 5 6 7 10 12 14 16 18 21 22 23 26 28 29 30 34 35 38 39 41 43 45 53 54 58 59 60 61 63 64 65 71 72 75 81 86 91 93 97 98 99)

# Start with a fresh output file
> output.log

# Loop over each index and run the Python script
for index in "${indices[@]}"; do
    echo "Running fuzz_single_question_single_model.py with question_index=$index" >> output.log
    python -u fuzz_single_question_single_model.py --question_index "$index" >> output.log 2>&1
done

echo "All runs completed!" >> output.log
