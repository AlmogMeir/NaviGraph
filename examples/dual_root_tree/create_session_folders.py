"""Create session folders in resources/prev_sessions for new recording dates.

Scans resources/maze_videos for <subject>_<YYYY>_<MM>_<DD>.mp4 files, matches each
to its <subject>_<YYYY>_<MM>_<DD>_corrected.h5 tracking in resources/maze_trackings,
and copies both into resources/prev_sessions/session_<subject>_<DD>_<MM>_<YYYY>/.

Dates that already have a session folder are skipped, as are dates with no
corrected tracking file yet.
"""

import argparse
import os
import re
import shutil
import sys

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
VIDEOS_DIR = os.path.join(RESOURCES, "maze_videos")
TRACKINGS_DIR = os.path.join(RESOURCES, "maze_trackings")
SESSIONS_DIR = os.path.join(RESOURCES, "prev_sessions")

VIDEO_PATTERN = re.compile(r"^(?P<subject>[^_]+)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})\.mp4$")


def find_candidates():
    """Return (session_name, video_path, tracking_path) for each new, complete date."""
    ready, missing_tracking, existing = [], [], []

    for filename in sorted(os.listdir(VIDEOS_DIR)):
        match = VIDEO_PATTERN.match(filename)
        if match is None:
            continue

        subject, year, month, day = match.group("subject", "year", "month", "day")
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

        ready.append((session_name, os.path.join(VIDEOS_DIR, filename), tracking_path))

    return ready, missing_tracking, existing


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

    ready, missing_tracking, existing = find_candidates()

    print(f"{len(existing)} session folder(s) already present, skipped.")
    for session_name, tracking_name in missing_tracking:
        print(f"SKIP {session_name}: no tracking file {tracking_name}")

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
