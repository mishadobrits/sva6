"""
This module contains algorithms that get moviepy_video_object and return a list of interesting parts
in format [[start_of_piece0, end_of_piece0], [start_of_piece1, end_of_piece1], ...
[start_of_piecen, end_of_piecen]]
All values should be positions in video in seconds.
All algorithms must be inherited from the base class 'SpeedUpAlgorithm.'
Currently, there are
 'VolumeThresholdAlgorithm(sound_threshold)'
 'WebRtcVADAlgorithm(aggressiveness)'
 'SileroVadAgorithm()'


"""
import math
import numpy as np
from .some_functions import str2error_message, save_audio_to_wav, WavSubclip, get_duration


def apply_min_silence_time_to_interesting_parts(min_silence_time_sec, interesting_parts):
    if not interesting_parts.size:
        return interesting_parts.reshape((-1, 2))
    max_time = interesting_parts[-1, 1]
    interesting_parts[:, 1] += min_silence_time_sec
    interesting_parts = union(interesting_parts)
    return np.minimum(interesting_parts, max_time)


class SpeedUpAlgorithm:
    """
    Base class for all Algorithms
    """
    def __init__(self):
        pass

    def get_interesting_parts(self, video_path: str):
        """
        All classes inherited from SpeedUpAlgorithm must overload get_interesting_parts method,
        because this method used by main.apply_calculated_interesting_to_video and
        main.process_one_video_in_computer

        :param moviepy_video: VideoClip
        :return: np.array of interesting parts in usual format
                (format in look settings.process_interestingpartsarray.__doc__)
        """
        msg = f"All classes inherited from {__class__} must overload get_loud_parts method"
        raise AttributeError(str2error_message(msg))

    def __str__(self):
        return f"{type(self).__name__}"


class WavSoundAlgorithm(SpeedUpAlgorithm):
    """
    The same as PiecemealBaseAlgorithm but for algorithms, that uses only sound.
    """
    def get_interesting_parts(self, video_path: str):
        wav_audio_path = save_audio_to_wav(video_path)
        wav_audio = WavSubclip(wav_audio_path)
        return self.get_interesting_parts_from_wav(wav_audio)

    def get_interesting_parts_from_wav(self, wav_audio: WavSubclip):
        msg = f"All classes inherited from {__class__} must overload" + \
              " get_interesting_parts_from_wav method"
        raise AttributeError(msg)


class PiecemealWavSoundAlgorithm(WavSoundAlgorithm):
    """
    The same as PiecemealBaseAlgorithm but for algorithms, that uses only sound.
    """
    def __init__(self, chunk_in_seconds: float = 60):
        self.chunk = chunk_in_seconds
        super(PiecemealWavSoundAlgorithm, self).__init__()

    def get_interesting_parts_from_wav(self, wav_audio):
        interesting_parts = []
        print(f"from {math.ceil(wav_audio.duration / self.chunk)}: ", end="")
        for start in np.arange(0, wav_audio.duration, self.chunk):
            print(round(start / self.chunk), end=", ")
            end = min(start + self.chunk, wav_audio.duration)
            wav_part = wav_audio.subclip(start, end)

            chunk_interesting_parts = np.array(self.get_interesting_parts_from_wav_part(wav_part))
            if chunk_interesting_parts.size:
                interesting_parts.append(start + chunk_interesting_parts)
        print()
        return np.vstack(interesting_parts)

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavSubclip):
        msg = f"All classes inherited from {__class__} must overload" + \
              " get_interesting_parts_from_wav method"
        raise AttributeError(msg)


class VolumeThresholdAlgorithm(PiecemealWavSoundAlgorithm):
    """
    Returns pieces where volume >= sound_threshold as interesting parts

    min_quiet_time - the program doesn't accelerate the
     first min_quiet_time seconds in each of boring piece.

    """
    def __init__(self,
                 sound_threshold: float,
                 min_quiet_time: float = 0.25,
                 chunk_in_seconds: float = 60):
        self.sound_threshold = sound_threshold
        self.min_q_time = min_quiet_time
        super(VolumeThresholdAlgorithm, self).__init__(chunk_in_seconds=chunk_in_seconds)

    def set_sound_threshold(self, value: float):
        self.sound_threshold = value

    def get_sound_threshold(self):
        """:returns sound_threshold: float"""
        return self.sound_threshold

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavSubclip):
        sound = np.abs(wav_audio_chunk.to_soundarray())
        sound = sound.max(axis=1).reshape(-1)
        sound = np.hstack([-1, sound, self.sound_threshold + 1, -1])

        is_voice = (sound > self.sound_threshold).astype(int)
        borders = is_voice[1:] - is_voice[:-1]
        begin_sound_indexes = np.arange(len(borders))[borders > 0]
        end_sound_indexes = np.arange(len(borders))[borders < 0]

        interesting_parts = np.vstack([begin_sound_indexes, end_sound_indexes])
        interesting_parts = interesting_parts.transpose((1, 0)) / wav_audio_chunk.fps
        return apply_min_silence_time_to_interesting_parts(self.min_q_time, interesting_parts)

    def __str__(self):
        return f"{type(self).__name__}(sound_threshold={self.get_sound_threshold()})"


"""
class EnergyThresholdAlgorithm(PiecemealSoundAlgorithm):
    def __init__(self, energy_threshold, audio_chunk):
        super(EnergyThresholdAlgorithm, self).__init__()
        pass  # todo
"""


class WebRtcVADAlgorithm(PiecemealWavSoundAlgorithm):
    """
    This algorithm selects speech from video using Voice Activity Detection (VAD)
    algorithm coded by google (link https://github.com/wiseman/py-webrtcvad)
    and returns them as interesting parts
    """
    def __init__(self,
                 aggressiveness: int = 1,
                 min_quiet_time: float = 0.25,
                 frame_duration: int = 30):
        """

        :param aggressiveness: parameter to VAD
        :param min_quiet_time: as usual
        :param frame_duration: must be 10, 20 or 30 - VAD parameter
        :param chunk_in_seconds:
        """
        try:
            import webrtcvad
        except ImportError as import_error:
            err = ImportError("WebRtcVADAlgorithm algorithm requires installed 'webrtcvad' module")
            raise err from import_error

        self.aggressiveness = None
        self.sample_rate = 48000
        self.frame_duration = frame_duration  # ms

        self.min_quiet_time = min_quiet_time
        self.set_aggressiveness(aggressiveness)

        super(WebRtcVADAlgorithm, self).__init__(chunk_in_seconds=60)

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavSubclip):
        from webrtcvad import Vad

        array, old_fps = wav_audio_chunk.to_soundarray(), wav_audio_chunk.fps
        index = np.arange(0, len(array), old_fps / self.sample_rate).astype(int)
        array = array[index]
        array = (abs(array) * 2 ** 16).astype("int16")

        prev_value = False
        chunk = 2 * int(self.sample_rate * self.frame_duration / 1000)
        array = np.hstack([[0] * chunk, array, [0] * 2 * chunk])
        begins_of_speech, ends_of_speech = [], []
        for i in range(0, len(array) - chunk, chunk):
            cur_sound = array[i: i + chunk]
            value = Vad(self.aggressiveness).is_speech(cur_sound.data, self.sample_rate)
            # I tried self.vad.is_speech(cur_sound.data, sample_rate),
            # but some how it isn't the same.
            if value and not prev_value:
                begins_of_speech.append(i / self.sample_rate)
            if not value and prev_value:
                ends_of_speech.append(i / self.sample_rate)
            prev_value = value

        interesting_parts = np.vstack([begins_of_speech, ends_of_speech]).transpose((1, 0))
        return apply_min_silence_time_to_interesting_parts(self.min_quiet_time,
                                                           interesting_parts)

    def get_aggressiveness(self):
        """:returns aggressiveness: int 0, 1, 2 or 3"""
        return self.aggressiveness

    def set_aggressiveness(self, aggressiveness: int):
        """sets aggressiveness = value"""
        assert aggressiveness in [0, 1, 2, 3],\
            f"aggressiveness must be 0, 1, 2 or 3. {aggressiveness} were given"
        self.aggressiveness = aggressiveness

    def __str__(self):
        return f"{type(self).__name__}({self.aggressiveness})"


class SileroVadAlgorithm(PiecemealWavSoundAlgorithm):
    """
    This algorithm selects speech from text using VAD algorithm
    from this (https://github.com/snakers4/silero-vad) project
    and returns them as interesting parts.
    """
    def __init__(self, *vad_args, onnx: bool = True, **vad_kwargs):
        try:
            import torch
        except ImportError:
            m = f"{__class__.__name__} requires torch, torchaudio and (onnxruntime if onnx=True in kwargs))"
            raise ImportError(m)
        super(SileroVadAlgorithm, self).__init__()

        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                model='silero_vad',
                                                force_reload=False,
                                                onnx=onnx)
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils

        self.vad_args, self.vad_kwargs = vad_args, vad_kwargs

    def get_interesting_parts_from_wav_part(self, wav_audio_chunk: WavSubclip):
        import torchaudio
        import torch
        available_rate = 16000
        sound = wav_audio_chunk.to_soundarray()[:, 0]
        sound = torch.tensor(sound, dtype=torch.float32)

        transform = torchaudio.transforms.Resample(orig_freq=wav_audio_chunk.fps,
                                                   new_freq=available_rate,
                                                   dtype=sound.dtype)
        sound = transform(sound)   # https://github.com/snakers4/silero-vad/blob/76687cbe25ffdf992ad824a36bfe73f6ae1afe72/utils_vad.py#L86

        dict_of_interesting_parts = self.get_speech_timestamps(
            sound,
            self.model,
            *self.vad_args,
            **self.vad_kwargs
        )
        # Todo I don't by what value we should divide timestamps. 16000 works.
        #  It should be replaced by an expression depending on vad_args, vad_kwargs
        #  https://t.me/silero_speech/1392
        list_of_interesting_parts = [[elem['start'] / available_rate, elem['end'] / available_rate]
                                     for elem in dict_of_interesting_parts]
        return np.array(list_of_interesting_parts)

    def __str__(self):
        answer = f"{type(self).__name__}("
        if self.vad_args:
            answer += f"vad_args={self.vad_args}, "
        if self.vad_kwargs:
            answer += f"vad_kwargs={self.vad_kwargs}, "
        if answer.endswith(", "):
            answer = answer[:-2]
        answer += ")"
        return answer


class AlgNot(SpeedUpAlgorithm):
    """
    Accepts algorithms as arguments and returns
     [pieces of parts that alg selects as boring] as interesting.
    Syntaxis:
        alg = AlgNot(alg)

    """
    def __init__(self, algorithm: SpeedUpAlgorithm):
        super(AlgNot, self).__init__()
        self.alg = algorithm

    def get_interesting_parts(self, video_path: str):
        return reverse(self.alg.get_interesting_parts(video_path), get_duration(video_path))

    def __str__(self):
        return f"(not {self.alg})"


class AlgAnd(SpeedUpAlgorithm):
    """
    Accepts algorithms as arguments and returns pieces of parts that all algorithms
     select as interesting.
    Syntaxis:
        alg = AlgAnd(alg1, alg2, alg3 ... algn)

    """
    def __init__(self, *algorithms):
        super(AlgAnd, self).__init__()
        self.algs = algorithms

    def get_interesting_parts(self, path: str):
        return intersection([alg.get_interesting_parts(path) for alg in self.algs])

    def __str__(self):
        result = " and ".join(map(str, self.algs))
        return f"({result})"


class AlgOr(SpeedUpAlgorithm):
    """
    Accepts algorithms as arguments and returns pieces of parts that at least one
     algorithm selects as interesting.
    Syntaxis:
        alg = AlgOr(alg1, alg2, alg3 ... algn)

    """
    def __init__(self, *algorithms):
        super(AlgOr, self).__init__()
        self.algs = algorithms

    def get_interesting_parts(self, video_path: str):
        return union([alg.get_interesting_parts() for alg in self.algs])

    def __str__(self):
        result = " or ".join(map(str, self.algs))
        return f"({result})"


def intersection_k(is_k_acceptable_func, *lists_of_interesting_parts):
    """
    Moments, where intersects at least k parts
    """
    borders = []
    for list_of_ip in lists_of_interesting_parts:
        for start, end in list_of_ip:
            borders.append([start, 0, 1])
            borders.append([end, 1, -1])

    borders.sort()
    rt = []
    current_parts = 0
    for time, priority, delta in borders:
        prev_current_parts = current_parts
        current_parts += delta
        if not is_k_acceptable_func(prev_current_parts) and is_k_acceptable_func(current_parts):
            rt.append([time])
        if is_k_acceptable_func(prev_current_parts) and not is_k_acceptable_func(current_parts):
            rt[-1].append(time)

    return np.array(rt)


def union(*lists_of_unions_of_segments):
    return intersection_k(lambda k: k>=1, *lists_of_unions_of_segments)


def intersection(*lists_of_unions_of_segments):
    n = len(lists_of_unions_of_segments)
    rt = intersection_k(lambda k: k==n, *lists_of_unions_of_segments)
    return [[a, b] for a, b in rt if b != a]


def reverse(list_of_interesting_parts, duration):
    whole_segment = [[0, duration]]
    reversed = intersection_k(lambda k: k == 1, *[whole_segment, list_of_interesting_parts])
    return intersection(*[reversed, whole_segment])


v1 = [[0, 3], [6, 8]]
v2 = [[2, 4], [5, 8]]
v3 = [[1, 5], [7, 9], [10, 11]]

"""
class _FakeDebugAlgorithm(SpeedUpAlgorithm):
    def __init__(self, interesting_parts):
        super(_FakeDebugAlgorithm, self).__init__()
        self.interesting_parts = np.array(interesting_parts, dtype="float64")

    def get_interesting_parts(self, video_path: str):
        duration = 10 ** 10 if not video_path else VideoFileClip(video_path).duration
        return np.minimum(self.interesting_parts, duration)

    def __str__(self):
        return f"{__class__.__name__}({self.interesting_parts.tolist()})"""
"""
class CropLongSounds(SpeedUpAlgorithm):
    def __init__(self, max_lenght_of_one_sound=0.05, threshold=0.995):
        self.step = max_lenght_of_one_sound
        self.threshold = threshold
        super(CropLongSounds, self).__init__()

    def get_interesting_parts(self, video_path: str):
        def cdot(a, b):
            if type(a) == type(b) == int:
                return a * b
            return (a * b).sum()

        def cos(a, b):
            if not cdot(a, a) * cdot(b, b):
                return 1
            return cdot(a, b) / (cdot(a, a) * cdot(b, b)) ** 0.5

        temporary_file_name = save_audio_to_wav(video_path)
        spec = 1
        interesting_parts = []
        with wave.open(temporary_file_name) as input_audio:
            duration = input_audio.getnframes() / input_audio.getframerate()
            for i in np.arange(0, duration - self.step, self.step):
                if random.random() < 0.01:
                    print(i)
                sound, rate = librosa.load(temporary_file_name, offset=i, duration=self.step)
                prev_spec = spec
                spec = librosa.feature.mfcc(sound, rate)
                if cos(spec, prev_spec) < self.threshold:
                    interesting_parts.append([i, i + self.step])
        interesting_parts.append([duration - self.step, self.step])
        return np.array(interesting_parts)

    def __str__(self):
        s = f"{__class__.__name__}(max_lenght_of_one_sound={self.step}, threshold={self.threshold})"
        return s"""