import shutil
import os

dirs = ['ja', 'en']

for dr in dirs:
    # Remove the existing directory
    if os.path.exists(f'./../docs/{dr}'):
        shutil.rmtree(f'./../docs/{dr}')

    # Copy the generated directory to docs
    shutil.copytree(f'generated/{dr}', f'./../docs/{dr}')