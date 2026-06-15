import os
import shutil
from src.interface.cli import CLI

# Force System FFmpeg for Colab
system_ffmpeg = "/usr/bin/ffmpeg"
if os.path.exists(system_ffmpeg):
    os.environ["IMAGEIO_FFMPEG_EXE"] = system_ffmpeg

if __name__ == "__main__":
    CLI().run()
