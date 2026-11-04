"""Stroke-based ("online") word detectors (issue #168).

Detectors that group the strokes of a page into words, instead of looking at a
rendering of the page. Base deps only (numpy, scikit-learn). The grouped
strokes are turned into word boxes that a vision-based recognizer such as
SimpleHTR then reads.
"""
