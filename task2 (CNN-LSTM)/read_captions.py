# Script to read and display first few lines of captions file
with open('data/captions.txt', 'r') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 9:  # Read first 10 lines
            break 