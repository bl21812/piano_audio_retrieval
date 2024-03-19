import sys
sys.path.append('Video-LLaMA')

import yaml
import os

with open('Video-LLaMA/environment.yml') as file:
    environment_data = yaml.safe_load(file)

for dep in environment_data['dependencies']:
    if isinstance(dep, dict):
        for lib in dep['pip']:
            os.system(f"pip install {lib}")
