"""Create session folders in resources/prev_sessions for new recording dates.

Scans resources/maze_videos for <subject>_<YYYY>_<MM>_<DD>.mp4 files, matches each
to its <subject>_<YYYY>_<MM>_<DD>_corrected.h5 tracking in resources/maze_trackings,
and copies both into resources/prev_sessions/session_<subject>_<DD>_<MM>_<YYYY>/.

Newer videos also carry a start time and a suffix, e.g.
3L_2025_12_08_14_43_53_stitched_blacked.mp4, while their tracking is still named by
date only. When a date has several videos (a restarted session), the one whose frame
count equals the tracking's row count is taken; if none matches, the date is skipped.

Dates that already have a session folder are skipped, as are dates with no
corrected tracking file yet.
"""

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict

import cv2
import pandas as pd

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
VIDEOS_DIR = os.path.join(RESOURCES, "maze_videos")
TRACKINGS_DIR = os.path.join(RESOURCES, "maze_trackings")
SESSIONS_DIR = os.path.join(RESOURCES, "prev_sessions")

VIDEO_PATTERN = re.compile(
    r"^(?P<subject>[^_]+)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
    r"(?:_\d{2}_\d{2}_\d{2})?(?:_[^.]*)?\.mp4$"
)


def video_frame_count(video_path):
    capture = cv2.VideoCapture(video_path)
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def pick_video(video_paths, tracking_path):
    """Return (video_path, None), or (None, reason) when no single video fits."""
    if len(video_paths) == 1:
        return video_paths[0], None

    rows = len(pd.read_hdf(tracking_path))
    matches = [path for path in video_paths if video_frame_count(path) == rows]
    if len(matches) == 1:
        return matches[0], None

    names = ", ".join(os.path.basename(path) for path in video_paths)
    return None, f"{len(matches)} of {len(video_paths)} videos ({names}) have {rows} frames like the tracking"


def find_candidates():
    """Return (session_name, video_path, tracking_path) for each new, complete date."""
    ready, missing_tracking, ambiguous, existing = [], [], [], []

    videos_by_date = defaultdict(list)
    for filename in sorted(os.listdir(VIDEOS_DIR)):
        match = VIDEO_PATTERN.match(filename)
        if match is None:
            continue
        key = match.group("subject", "year", "month", "day")
        videos_by_date[key].append(os.path.join(VIDEOS_DIR, filename))

    for (subject, year, month, day), video_paths in sorted(videos_by_date.items()):
        session_name = f"session_{subject}_{day}_{month}_{year}"
        session_path = os.path.join(SESSIONS_DIR, session_name)

        if os.path.isdir(session_path):
            existing.append(session_name)
            continue

        tracking_name = f"{subject}_{year}_{month}_{day}_corrected.h5"
        tracking_path = os.path.join(TRACKINGS_DIR, tracking_name)
        if not os.path.isfile(tracking_path):
            missing_tracking.append((session_name, tracking_name))
            continue

        video_path, reason = pick_video(video_paths, tracking_path)
        if video_path is None:
            ambiguous.append((session_name, reason))
            continue

        ready.append((session_name, video_path, tracking_path))

    return ready, missing_tracking, ambiguous, existing


def transfer(src, dst, link):
    if link:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="list what would be created without writing anything")
    parser.add_argument("--link", action="store_true", help="hard-link the files instead of copying (same filesystem only)")
    args = parser.parse_args()

    for directory in (VIDEOS_DIR, TRACKINGS_DIR, SESSIONS_DIR):
        if not os.path.isdir(directory):
            sys.exit(f"Missing directory: {directory}")

    ready, missing_tracking, ambiguous, existing = find_candidates()

    print(f"{len(existing)} session folder(s) already present, skipped.")
    for session_name, tracking_name in missing_tracking:
        print(f"SKIP {session_name}: no tracking file {tracking_name}")
    for session_name, reason in ambiguous:
        print(f"SKIP {session_name}: can't tell which video the tracking belongs to; {reason}")

    if not ready:
        print("Nothing to create.")
        return

    verb = "Would create" if args.dry_run else "Creating"
    print(f"\n{verb} {len(ready)} session folder(s):")

    for session_name, video_path, tracking_path in ready:
        print(f"  {session_name}")
        print(f"    <- {os.path.basename(video_path)}")
        print(f"    <- {os.path.basename(tracking_path)}")
        if args.dry_run:
            continue

        session_path = os.path.join(SESSIONS_DIR, session_name)
        os.mkdir(session_path)
        for src in (video_path, tracking_path):
            transfer(src, os.path.join(session_path, os.path.basename(src)), args.link)

    print(f"\nDone. {len(ready)} session folder(s) {'planned' if args.dry_run else 'created'}.")


if __name__ == "__main__":
    main()
