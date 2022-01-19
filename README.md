# #todo normal description

##Usage
```python
from sva6.apply_timecodes_to_video import apply_v1_timecodes_to_video_2vfr
from sva6.speed_up import VolumeThresholdAlgorithm, SileroVadAlgorithm, intersection

video_path = r"C:\Users\m\Downloads\Sites-Buffers\video_with_sound_cutten.mp4"
interesting_parts1 = VolumeThresholdAlgorithm(0.02).get_interesting_parts(video_path)
interesting_parts2 = SileroVadAlgorithm(threshold=0.45).get_interesting_parts(video_path)
ip = interesting_parts = intersection(interesting_parts1, interesting_parts2)

v1timecodes = []
for i in range(len(interesting_parts) - 1):
    v1timecodes.append([ip[i][0], ip[i][1], 1])
    v1timecodes.append([ip[i][1], ip[i + 1][0], 6])  # ip[i + 1][0] --> min(ip[i + 1][0], ip[i][1] + 2)
v1timecodes.append([ip[-1][0], ip[-1][1], 1])

output_path = r"C:\Users\m\Downloads\Sites-Buffers\video_with_sound_cutten-sva.mp4"
apply_v1_timecodes_to_video_2vfr(v1timecodes, video_path, output_path, last_number_in_v1_represent="speed")
```

