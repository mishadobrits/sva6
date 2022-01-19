import itertools
import math
import os
import shutil
from tempfile import gettempdir
from wave import Wave_read
import numpy as np
import wavio
from moviepy.video.io.VideoFileClip import VideoFileClip
from .ffmpeg_caller import FFMPEGCaller
from .globals import TEMPORARY_DIRECTORY_PREFIX


def v1timecodes_to_v2timecodes(v1timecodes, video_fps, length_of_video, default_output_fps=10 ** 6):
    """

    :param v1timecodes: timecodes in v1format:
        [[start0, end0, fps0], [start1, end1, fps1], ... [start_i, end_i, fps_i]]
         (same as save_timecodes_to_v1_file)
        where start and end in seconds, fps in frames per second
    :return: v2timecodes: timecodes in v2format:
        [timecode_of_0_frame_in_ms, timecode_of_1st_frame_in_ms, ... timecode_of_nth_frame_in_ms]
    """

    default_freq = 1 / default_output_fps / video_fps
    time_between_neighbour_frames = default_freq * np.ones(length_of_video, dtype=np.float64)
    for elem in v1timecodes:

        start_t, end_t = elem[0] * video_fps, elem[1] * video_fps
        # todo begin kostil
        start_t, end_t = min(start_t, length_of_video - 1), min(end_t, length_of_video - 1)
        start_i, end_i = math.floor(start_t), math.floor(end_t) + 1

        X = time_between_neighbour_frames[start_i: end_i]
        if not X.size:
            continue
            # print(start_t, end_t, start_i, end_i)
        X += 1 / elem[2] * (end_t - start_t) / (end_i - start_i)
        # print((end_i - start_t) / elem[2] - sum(X))
        # end kostil


        """
        addition = 1 / elem[2] - default_freq
        # print(start_t, end_t, addition, ":")
        ceil_start_t, floor_end_t = math.ceil(start_t), math.floor(end_t)
        # print(f"  [{ceil_start_t}-{floor_end_t}] += {addition}")
        # print(f"  [{ceil_start_t - 1}] += {addition} * {(ceil_start_t - start_t)}")
        # print(f"  [{floor_end_t}] += {addition} * {(end_t - floor_end_t)}")
        time_between_neighbour_frames[ceil_start_t: floor_end_t] += addition
        time_between_neighbour_frames[ceil_start_t - 1] += addition * (ceil_start_t - start_t)
        time_between_neighbour_frames[floor_end_t] += addition * (end_t - floor_end_t)
        # """

    timecodes = cumsum(time_between_neighbour_frames)
    return timecodes


def cumsum(n1array):
    """
    np.nancumsum works wrong for me, so I wrote equivalent function
    :param n1array:
    :return: n1array of cumulative sums
    """
    accumalated_iter = itertools.accumulate(n1array.tolist())
    return np.array(list(itertools.chain([0], accumalated_iter)))


def save_v2_timecodes_to_file(filepath, timecodes):
    """
    :param filepath: path to file for saving
    :param timecodes: list of timecodes of each frame in format
            [timecode_of_0_frame_in_ms, timecode_of_1_frame_in_ms, ... timecode_of_i_frame_in_ms]
    :return: file object (closed)
    """

    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    str_timecodes = [format(elem * 1000, "f") for elem in timecodes]
    for i, (elem, next_elem) in enumerate(pairwise(str_timecodes)):
        if float(elem) >= float(next_elem):
            str_timecodes[i + 1] += str(i).rjust(10, "0")
    with open(filepath, "w") as file:
        file.write("# timestamp format v2\n")
        file.write("\n".join(str_timecodes))
    return file


def save_v1_timecodes_to_file(filepath, timecodes, videos_fps, default_fps=10 ** 10):
    """

    :param filepath: path of the file for saving
    :param timecodes: timecodes in format
        [[start0, end0, fps0], [start1, end1, fps1], ... [start_i, end_i, fps_i]]
    :param videos_fps: float fps of video
    :param default_fps: fps of uncovered pieces
    :return: closed file object in which timecodes saved
    """
    with open(filepath, "w") as file:
        file.write("# timecode format v1\n")
        file.write(f"assume {default_fps}\n")
        for elem in timecodes:
            elem = [int(elem[0] * videos_fps), int(elem[1] * videos_fps), elem[2]]
            elem = [str(n) for n in elem]
            file.write(",".join(elem) + "\n")
    return file


def read_bytes_from_wave(waveread_obj: Wave_read, start_sec: float, end_sec: float):
    """
    Reades bytes from wav file 'waveread_obj' from start_sec up to end_sec.

    :param waveread_obj: Wave_read
    :param start_sec: float
    :param end_sec: float
    :return: rt_bytes: bytes: read bytes
    """
    previous_pos, framerate = waveread_obj.tell(), waveread_obj.getframerate()

    start_pos = min(waveread_obj.getnframes(), math.ceil(framerate * start_sec))
    end_pos = min(waveread_obj.getnframes(), math.ceil(framerate * end_sec))

    waveread_obj.setpos(start_pos)
    rt_bytes = waveread_obj.readframes(end_pos - start_pos)
    waveread_obj.setpos(previous_pos)

    return rt_bytes


def delete_all_sva_temporary_objects():
    """
    When process_one_video_in_computer or apply_calculated_interesting_to_video
    creates temporary directory or temporary file its name starts with
    TEMPORARY_DIRECTORY_PREFIX="SVA4_" for easy identification.
    If user terminates process function doesn't delete directory, cause of it terminated.
    So, function delete_all_tempories_sva4_directories deletes all directories and files which
    marked with prefix TEMPORARY_DIRECTORY_PREFIX
    :return: None
    """
    temp_dirs = [f for f in os.scandir(gettempdir()) if (f.is_dir() or f.is_file())]
    sva4_temp_dirs = [f for f in temp_dirs if f.name.startswith(TEMPORARY_DIRECTORY_PREFIX)]
    for temp_path in sva4_temp_dirs:
        full_path = os.path.join(gettempdir(), temp_path.path)
        print(f"Deleting {full_path}")
        if temp_path.is_dir():
            shutil.rmtree(full_path)
        elif temp_path.is_file():
            os.remove(full_path)


class WavSubclip:
    def __init__(self, path: str, start: float = 0, end: float = 10 ** 10):
        self.path = path
        self.start = start
        with Wave_read(path) as wav_file:
            self.sample_width = wav_file.getsampwidth()
            self.nchannels = wav_file.getnchannels()
            self.fps = wav_file.getframerate()
            self.duration = wav_file.getnframes() / self.fps
            self.end = min(end, self.duration)

    def subclip(self, start, end):
        return WavSubclip(self.path, self.start + start, self.start + end)

    def to_soundarray(self):
        sample_width = self.sample_width
        sample_range = wavio._sampwidth_ranges[sample_width]
        read_bytes = read_bytes_from_wave(Wave_read(self.path), self.start, self.end)
        array = wavio._wav2array(self.nchannels, sample_width, read_bytes).astype("float64")

        # print(array, array.min(), array.max(), sample_width, sample_range)
        # breakpoint()
        array = (array - sample_range[0]) / (sample_range[1] - sample_range[0])
        # print("array", array,)
        return 2 * array - 1  # fit to [-1, 1] range

    def read_part(self, start, end):
        return self.subclip(start, end).to_soundarray()


def save_audio_to_wav(input_video_path):
    """
    Saves videos audio to wav and returns its path
    :param input_video_path:
    :return: path od audio

    """
    ffmpeg = FFMPEGCaller(overwrite_force=False, hide_output=True, print_command=True)

    input_video_path = os.path.abspath(input_video_path)
    filename = TEMPORARY_DIRECTORY_PREFIX + str(hash(input_video_path)) + ".wav"
    filepath = os.path.join(gettempdir(), filename)

    ffmpeg(f"-i {input_video_path} {filepath}")
    return filepath


def str2error_message(msg):
    """Deletes \n from msg and replace ' '*n -> ' '"""
    return " ".join(list(msg.replace("\n", " ").split()))


def get_duration(video_path: str):
    return VideoFileClip(video_path).duration


def get_fps(video_path: str):
    return VideoFileClip(video_path).fps