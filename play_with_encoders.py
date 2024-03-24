import sys
sys.path.append('Video-LLaMA')

import video_llama.models as video_llama

from models import load_video_llama_modules

# zoo = models.ModelZoo()
# print(zoo)

modules = load_video_llama_modules()

'''for module_name, module in modules.items():
    print(module)
    print(module_name)
    input()'''

# audio preprocessing - take spectrograms of consecutive 2-sec audio clips
    # as basically a time series of images
