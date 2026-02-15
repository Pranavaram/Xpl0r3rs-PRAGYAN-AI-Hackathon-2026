import re

file_path = '/Users/hansikanm/Downloads/Frontend/src/data/dummyData.ts'

with open(file_path, 'r') as f:
    content = f.read()

lines = content.split('\n')
new_lines = []
for line in lines:
    if 'vitals: {' in line:
        # First remove any existing respiratoryRate
        line = line.replace(', respiratoryRate: 18', '')
        line = line.replace('respiratoryRate: 18, ', '')
        # Now add it back ensuring it's there exactly once at the end
        # We assume the line ends with something like ' }' or just '}'
        # This simple replacement works for the one-line format in dummyData.ts
        if line.strip().endswith('},'):
           line = line.replace('},', ', respiratoryRate: 18 },')
        elif line.strip().endswith('}'):
           line = line.replace('}', ', respiratoryRate: 18 }')
    new_lines.append(line)

final_content = '\n'.join(new_lines)
# Clean up any double commas caused by replace
final_content = final_content.replace(',,', ',')

with open(file_path, 'w') as f:
    f.write(final_content)
