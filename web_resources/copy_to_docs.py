import shutil
import os
from pathlib import Path
from check_translation_notices import check_translation_notices


check_translation_notices(Path(__file__).absolute().parent, include_generated=True)

dirs = ['ja', 'en']

for dr in dirs:
    # Remove the existing directory
    if os.path.exists(f'./../docs/{dr}'):
        shutil.rmtree(f'./../docs/{dr}')

    # Copy the generated directory to docs
    shutil.copytree(f'generated/{dr}', f'./../docs/{dr}')
