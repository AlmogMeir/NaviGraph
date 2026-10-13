"""Fill in missing session images by grabbing a frame out of the session video.

Every session needs a still image of the maze: it is what the mapping GUI and the
calibration plugin are pointed at (`setup.spatial_image_for_calibration`), and what
render_all_mappings.py draws each mapping on top of. The image is just a frame of
that session's own video, so any date with a video in resources/maze_videos but no
resources/maze_images/<subject>_<YYYY>_<MM>_<DD>.png can have one extracted here.
Videos whose name also carries a start time (<subject>_<date>_<HH>_<MM>_<SS>_*.mp4)
get <subject>_<date>_<HH>_<MM>_<SS>.png, so same-day sessions stay distinct.

The frame index matters: frame 0 is often mid-trigger or before the lights settle,
so the default is frame 1000 (25 s in at 40 FPS). A video shorter than that falls
back to its last frame rather than being skipped, since a short session still needs
an image.

    python extract_missing_session_images.py              # extract every missing image
    python extract_missing_session_images.py --dry-run    # just report what is missing
    python extract_missing_session_images.py --frame 2000 # take a later frame
    python extract_missing_session_images.py --only NPC4  # one subject
    python extract_missing_session_images.py --force      # re-extract images that exist

Existing images are never overwritten unless --force is passed.
"""

import argparse
import re
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"
VIDEOS_DIR = RESOURCES / "maze_videos"
IMAGES_DIR = RESOURCES / "maze_images"

DEFAULT_FRAME = 1000

# Older recordings are named <subject>_YYYY_MM_DD.mp4; newer ones carry the start
# time and a processing suffix, e.g. 3L_2025_11_17_19_40_05_stitched_blacked.mp4.
VIDEO_PATTERN = re.compile(
    r"^(?P<subject>[^_]+)_(?P<date>\d{4}_\d{2}_\d{2})"
    r"(?:_(?P<time>\d{2}_\d{2}_\d{2}))?(?:_[^.]*)?\.mp4$"
)


def session_name_for(match):
    """Image/session name for a video: subject + date, plus the start time when the
    video has one (some subjects have two sessions on the same day)."""
    name = f"{match.group('subject')}_{match.group('date')}"
    if match.group("time"):
        name += f"_{match.group('time')}"
    return name


def find_candidates(only=None, force=False):
    """Return (sessions needing an image, sessions already covered).

    Each candidate is (session_name, video_path, image_path).
    """
    missing, covered = [], []

    for video_path in sorted(VIDEOS_DIR.iterdir()):
        match = VIDEO_PATTERN.match(video_path.name)
        if match is None:
            continue
        if only is not None and match.group("subject") != only:
            continue

        session_name = session_name_for(match)
        image_path = IMAGES_DIR / f"{session_name}.png"

        if image_path.is_file() and not force:
            covered.append(session_name)
        else:
            missing.append((session_name, video_path, image_path))

    return missing, covered


def grab_frame(video_path, frame_index):
    """Return (frame, index actually read) for this video, or (None, reason)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None, f"could not open {video_path.name}"

    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # CAP_PROP_FRAME_COUNT is an estimate on some encodes, so treat it as a
        # hint for the seek target and let the read decide what actually exists.
        wanted = frame_index if total <= 0 or frame_index < total else max(total - 1, 0)

        capture.set(cv2.CAP_PROP_POS_FRAMES, wanted)
        ok, frame = capture.read()

        while not ok and wanted > 0:
            # Seek overshot the real end of the stream: walk back until a frame reads.
            wanted = wanted // 2
            capture.set(cv2.CAP_PROP_POS_FRAMES, wanted)
            ok, frame = capture.read()

        if not ok:
            return None, f"no readable frame in {video_path.name}"
        return frame, wanted
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME,
                        help=f"frame index to extract (default {DEFAULT_FRAME})")
    parser.add_argument("--only", metavar="SUBJECT", help="restrict to one subject, e.g. NPC4")
    parser.add_argument("--dry-run", action="store_true", help="list what would be extracted without writing anything")
    parser.add_argument("--force", action="store_true", help="re-extract sessions that already have an image")
    args = parser.parse_args()

    if args.frame < 0:
        sys.exit("--frame must be >= 0")

    for directory in (VIDEOS_DIR, IMAGES_DIR):
        if not directory.is_dir():
            sys.exit(f"Missing directory: {directory}")

    missing, covered = find_candidates(only=args.only, force=args.force)

    print(f"{len(covered)} session(s) already have an image, skipped.")

    if not missing:
        print("Nothing to extract.")
        return

    verb = "Would extract" if args.dry_run else "Extracting"
    print(f"\n{verb} {len(missing)} session image(s) at frame {args.frame}:")

    written, failed = 0, []
    for session_name, video_path, image_path in missing:
        print(f"  {session_name}")
        print(f"    <- {video_path.name}")
        if args.dry_run:
            continue

        frame, info = grab_frame(video_path, args.frame)
        if frame is None:
            print(f"    FAILED: {info}")
            failed.append(session_name)
            continue

        if info != args.frame:
            print(f"    note: video is shorter than frame {args.frame}, used frame {info} instead")

        if not cv2.imwrite(str(image_path), frame):
            print(f"    FAILED: could not write {image_path}")
            failed.append(session_name)
            continue

        print(f"    -> {image_path.relative_to(HERE)} ({frame.shape[1]}x{frame.shape[0]})")
        written += 1

    if args.dry_run:
        print(f"\nDone. {len(missing)} session image(s) planned.")
        return

    print(f"\nDone. {written} session image(s) written.")
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
