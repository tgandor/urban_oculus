# mogrify to successive qualities 50 imgs at a time
# works best with val2017 ;)

import glob
import os

imgs = sorted(glob.glob("*.jpg"))

for q, img in zip((i for i in range(1, 101) for _ in range(50)), imgs):
    print(img)
    os.system(f"mogrify -quality {q} {img}")
    os.rename(img, f"{q:03d}_{img}")

