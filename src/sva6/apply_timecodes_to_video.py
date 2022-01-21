# r"""
import os
import shutil
import sys

from typing import List, Callable, Any
from .globals import TEMPORARY_DIRECTORY_PREFIX, PATH_TO_MKVMERGE

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal
from wave import Wave_write, Wave_read
from tempfile import mkdtemp
from moviepy.video.io.VideoFileClip import VideoFileClip
from .ffmpeg_caller import FFMPEGCaller
from .some_functions import v1timecodes_to_v2timecodes, save_v2_timecodes_to_file, \
    read_bytes_from_wave, delete_all_sva_temporary_objects
from .time_stretch.time_stretch import stretch
import wavio
import numpy as np


def apply_v1_timecodes_to_audio(
        startt_endt_speed_list,
        audio_path: str,
        output_path: str,
        speed_up_function: Callable = stretch,
):
    with Wave_write(output_path) as output_wav, Wave_read(audio_path) as input_wav:
        output_wav.setnchannels(1)
        nchannels = input_wav.getnchannels()
        framerate = input_wav.getframerate()
        sampwidth = input_wav.getsampwidth()
        output_wav.setframerate(framerate)
        output_wav.setsampwidth(sampwidth)

        for (start_time, end_time, speed) in startt_endt_speed_list:
            read_bytes = read_bytes_from_wave(input_wav, start_time, end_time)
            audio = wavio._wav2array(nchannels, sampwidth, read_bytes)

            if not audio.size:
                continue
            x_streched = speed_up_function(audio[:, 0].astype(np.float64), speed).astype(audio.dtype)
            output_wav.writeframes(wavio._array2wav(x_streched, input_wav.getsampwidth()))


def apply_v1_timecodes_to_video_2vfr(
        v1timecodes,
        videopath: str,
        output_path: str,
        ffmpeg_caller: FFMPEGCaller = FFMPEGCaller(hide_output=True),
        speed_up_function: Callable=stretch,
        last_number_in_v1_represent: Literal["fps", "speed"] = "fps"
):
    working_directory_path = mkdtemp(prefix=TEMPORARY_DIRECTORY_PREFIX)

    def tpath(filename):
        """
        returns the absolute path for a file with name filename in folder working_directory_path
        """
        return os.path.join(working_directory_path, filename)

    video = VideoFileClip(videopath)
    assert last_number_in_v1_represent in ["fps", "speed"]
    if last_number_in_v1_represent == "fps":
        startt_endt_speed = [[start_t, end_t, current_fps / video.fps] for start_t, end_t, current_fps in v1timecodes]
    else:
        startt_endt_speed = [elem[:] for elem in v1timecodes]
        v1timecodes = [[start_t, end_t, current_fps * video.fps] for start_t, end_t, current_fps in startt_endt_speed]

    origin_audio_path, output_audio_path = tpath("wav_origin.wav"), tpath("wav_processed.wav")
    ffmpeg_caller(f"-i {videopath} -vn {origin_audio_path}")
    apply_v1_timecodes_to_audio(
        startt_endt_speed,
        origin_audio_path,
        output_audio_path,
        speed_up_function
    )

    tempory_video_path = tpath("tempory_video.mkv")

    v2timecodes_path = tpath("timecodes.v2")
    v2timecodes = v1timecodes_to_v2timecodes(v1timecodes, video.fps, video.reader.nframes)
    save_v2_timecodes_to_file(v2timecodes_path, v2timecodes)
    os.system(f"{PATH_TO_MKVMERGE} -o {tempory_video_path} --timestamps 0:{v2timecodes_path} -A {videopath} {output_audio_path}")

    shutil.copy(tempory_video_path, output_path)
    delete_all_sva_temporary_objects()
