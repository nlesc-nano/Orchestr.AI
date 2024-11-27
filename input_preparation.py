#!/usr/bin/env python

import numpy as np
import os
import re

# INPUT
# Processing the xyz files
# Specify the positions xyz file and read its contents
positions_file = 'CsPbBr3_2.4nm_PBE_MD_NVE_2.5fs-pos-1.xyz'

with open(positions_file, 'r') as file:
    lines = file.readlines()

# Specify the forces xyz file and read its contents
forces_file = 'CsPbBr3_2.4nm_PBE_MD_NVE_2.5fs-frc-1.xyz'

with open(forces_file, 'r') as file2:
    lines2 = file2.readlines()

# Specify the output xyz files
temp_file = 'postemp.xyz'
temp_file2 = 'fortemp.xyz'
output_file = 'cspbbr3.xyz'
# INPUT

# We remove the stuff from the header of each step and we maintain only the energies

# Process each line
modified_lines = []
for line in lines:
    if line.startswith(' i'):
        # Use regular expression to keep only the last floating-point number,
        # include the "-" sign, and add a line break after it
        modified_line = re.sub(r'.*?(-?\d+\.\d+)\D*$', r'\1\n', line)
        modified_lines.append(modified_line)
    else:
        modified_lines.append(line)

# Write the modified lines to the output file
with open(temp_file, 'w') as file:
    file.writelines(modified_lines)


# In[4]:


# For the forces, we also remove the stuff from the header of each step and maintain only the energies.
# This file will be used later for combining both

# Process each line
modified_lines = []
for line in lines2:
    if line.startswith(' i'):
        # Use regular expression to keep only the last floating-point number,
        # include the "-" sign, and add a line break after it
        modified_line = re.sub(r'.*?(-?\d+\.\d+)\D*$', r'\1\n', line)
        modified_lines.append(modified_line)
    else:
        modified_lines.append(line)

# Write the modified lines to the output file
with open(temp_file2, 'w') as file:
    file.writelines(modified_lines)


# In[5]:


# Function for finding the difference lines
def find_difference_index(line1, line2):
    min_length = min(len(line1), len(line2))
    for i in range(min_length):
        if line1[i] != line2[i]:
            return i
    return min_length

# Function that puts the forces after the positions
def combine_files(file1_path, file2_path, output_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    combined_lines = []
    for line1, line2 in zip(lines1, lines2):
        diff_index = find_difference_index(line1, line2)
        combined_line = line1.strip() + ' ' + line2[diff_index:].strip()
        combined_lines.append(combined_line)

    with open(output_path, 'w') as output_file:
        output_file.write('\n'.join(combined_lines))


# In[6]:


# We combine positions (temp_file) with forces (temp_file2)
combine_files(temp_file, temp_file2, output_file)


# In[7]:


# We clean the temporary files that were created and are not useful anymore
if os.path.exists('postemp.xyz'):
    os.remove('postemp.xyz')
elif os.path.exists('fortemp.xyz'):
    os.remove('fortemp.xyz')


# In[8]:


# Generate npz file'
if os.path.exists('cspbbr3.npz'):
    os.remove('cspbbr3.npz')

get_ipython().run_line_magic('run', 'xyztonpz.py cspbbr3.xyz')


# In[ ]:




