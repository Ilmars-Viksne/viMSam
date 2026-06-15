
# ==========================================
# 🚀 MASTER TEST SUITE
# ==========================================
import os
import shutil
import numpy as np
import imageio.v3 as imageio
from skimage.data import binary_blobs
from skimage.draw import disk
from IPython.display import Image, display, Video

def run_master_tsuit_2610():
  # --- 1. CONFIGURATION ---

  BASE_DIR = "segmenter_app"
  DATA_DIR = f"{BASE_DIR}/data"

  # Define Paths
  IMG_IN  = f"{DATA_DIR}/input/images"
  VID_IN  = f"{DATA_DIR}/input/videos"
  IMG_OUT = f"{DATA_DIR}/output/images"
  VID_OUT = f"{DATA_DIR}/output/videos"

  TEST_IMG   = f"{IMG_IN}/test_image.tif"
  BATCH_DIR  = f"{IMG_IN}/input_images"
  TEST_VIDEO = f"{VID_IN}/moving_cell.mp4"

  # Clean and Recreate Directories
  if os.path.abspath(DATA_DIR) not in ("/", "C:\\", "D:\\"):
      if os.path.exists(DATA_DIR):
          shutil.rmtree(DATA_DIR)

  for d in [IMG_IN, VID_IN, IMG_OUT, VID_OUT]:
      os.makedirs(d, exist_ok=True)

  os.makedirs(BATCH_DIR, exist_ok=True)

  # --- 2. DATA GENERATION FUNCTIONS ---

  def generate_synthetic_image(filename, seed, fixed_targets=None, random_targets_count=0):
      np.random.seed(seed)
      texture = binary_blobs(length=512, blob_size_fraction=0.05, volume_fraction=0.2)
      data = np.where(texture, 120, 30).astype(np.uint8)
      targets = []
      if fixed_targets: targets.extend(fixed_targets)
      if random_targets_count > 0:
          for _ in range(random_targets_count):
              targets.append(np.random.randint(50, 462, size=2))
      for (r, c) in targets:
          rr, cc = disk((r, c), 25)
          rr, cc = np.clip(rr, 0, 511), np.clip(cc, 0, 511)
          data[rr, cc] = 120
      noise = np.random.randint(0, 20, (512, 512), dtype=np.uint8)
      data = np.clip(data + noise, 0, 255).astype(np.uint8)
      imageio.imwrite(filename, data)
      return data

  print("⚙️ Generating Test Data...")

  # A. Main Test Image
  generate_synthetic_image(TEST_IMG, seed=42, fixed_targets=[(100, 150), (400, 200)])

  # B. Batch Images
  for i in range(3):
      generate_synthetic_image(f"{BATCH_DIR}/sample_{i}.tif", seed=100+i, random_targets_count=np.random.randint(2, 5))

  # C. Video Generation
  np.random.seed(999)
  bg_texture = binary_blobs(length=512, blob_size_fraction=0.05, volume_fraction=0.2)
  bg_base = np.where(bg_texture, 120, 30).astype(np.uint8)
  frames = []
  for i in range(20):
      frame = bg_base.copy()
      rr, cc = disk((100 + i*10, 150 + i*10), 30)
      frame[rr, cc] = 120
      frame = np.clip(frame + np.random.randint(0, 20, (512, 512)), 0, 255).astype(np.uint8)
      frames.append(frame)
  imageio.imwrite(TEST_VIDEO, frames, fps=5)
  print("✅ Data Ready.\n")

  # ==========================================
  # 🧪 EXECUTE TESTS
  # ==========================================

  # TEST 1: Single Image (Automatic) - Default behavior (Mask only)
  print("🧪 TEST 1: Single Image (Automatic)")
  #!python segmenter_app/main.py --input "{TEST_IMG}" --out "{IMG_OUT}/res_auto.png" --workflow single
  os.system(f'python segmenter_app/main.py --input "{TEST_IMG}" --out "{IMG_OUT}/res_auto.png" --workflow single')

  display(Image(f"{IMG_OUT}/res_auto.png", width=600))

  # TEST 2A: Single Image (Prompt A)
  print("\n🧪 TEST 2A: Single Image (Prompt: 150,100)")
  #!python segmenter_app/main.py --input "{TEST_IMG}" --out "{IMG_OUT}/res_point_a.png" --workflow single --points "150,100" --format csv
  os.system(f'python segmenter_app/main.py --input "{TEST_IMG}" --out "{IMG_OUT}/res_point_a.png" --workflow single --points "150,100" --format csv')
  display(Image(f"{IMG_OUT}/res_point_a.png", width=600))

  # TEST 2C: Single Image (Multi-Prompt) WITH COMBINED OUTPUT
  print("\n🧪 TEST 2C: Single Image (Multi-Prompt) [Combined View]")
  # Added --save_combined here
  os.system(f'python segmenter_app/main.py --input "{TEST_IMG}" --out "{IMG_OUT}/res_point_multi.png" --workflow single --points "150,100 200,400" --show_prompts --save_combined --format csv')
  '''!python segmenter_app/main.py --input "{TEST_IMG}" --out "{IMG_OUT}/res_point_multi.png" \
      --workflow single \
      --points "150,100 200,400" \
      --show_prompts \
      --save_combined \
      --format csv'''

  display(Image(f"{IMG_OUT}/res_point_multi.png", width=800))

  # TEST 4: Video Tracking WITH COMBINED OUTPUT

  # --- RUN A: BOX ---
  print("\n🧪 TEST 4A: Video Tracking (BOX) [Combined View]")
  out_box = f"{VID_OUT}/video_box"
  # Added --save_combined here
  os.system(f'python segmenter_app/main.py --input "{TEST_VIDEO}" --out "{out_box}" --workflow video --points "150,100" --tracking_method box --show_prompts --save_combined --format csv')
  '''!python segmenter_app/main.py --input "{TEST_VIDEO}" --out "{out_box}" \
      --workflow video \
      --points "150,100" \
      --tracking_method box \
      --show_prompts \
      --save_combined \
      --format csv'''

  if os.path.exists(f"{out_box}/result_video.mp4"):
      print("📺 Result (BOX Method):")
      display(Video(f"{out_box}/result_video.mp4", embed=True, width=600))

  # --- RUN B: CENTROID ---
  print("\n🧪 TEST 4B: Video Tracking (CENTROID) [Combined View]")
  out_centroid = f"{VID_OUT}/video_centroid"
  # Added --save_combined here
  os.system(f'python segmenter_app/main.py --input "{TEST_VIDEO}" --out "{out_centroid}" --workflow video --points "150,100" --tracking_method centroid --show_prompts --save_combined --format json')
  '''!python segmenter_app/main.py --input "{TEST_VIDEO}" --out "{out_centroid}" \
      --workflow video \
      --points "150,100" \
      --tracking_method centroid \
      --show_prompts \
      --save_combined \
      --format json'''

  if os.path.exists(f"{out_centroid}/result_video.mp4"):
      print("📺 Result (CENTROID Method):")
      display(Video(f"{out_centroid}/result_video.mp4", embed=True, width=600))

  # --- RUN C: POLE OF INACCESSIBILITY ---
  print("\n🧪 TEST 4C: Video Tracking (POLE) [Combined View]")
  out_pole = f"{VID_OUT}/video_pole"
  # Added --save_combined here
  os.system(f'python segmenter_app/main.py --input "{TEST_VIDEO}" --out "{out_pole}" --workflow video --points "150,100" --tracking_method pole --show_prompts --save_combined --format csv')
  '''!python segmenter_app/main.py \
      --input "{TEST_VIDEO}" \
      --out "{out_pole}" \
      --workflow video \
      --points "150,100" \
      --tracking_method pole \
      --show_prompts \
      --save_combined \
      --format csv'''

  if os.path.exists(f"{out_pole}/result_video.mp4"):
      print("📺 Result (POLE Method):")
      display(Video(f"{out_pole}/result_video.mp4", embed=True, width=600))
