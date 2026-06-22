"""
Extract frames from all videos in a directory for YOLO training data.

Extracts frames at a configurable interval to avoid near-duplicate images.
Supports .mov, .mp4, .avi, and other formats OpenCV can read.

Usage:
    python extract_frames.py --dir ./videos
    python extract_frames.py --dir ./videos --interval 0.5 --output ./frames
    python extract_frames.py --dir ./videos --interval 1.0 --max_frames 100
    python extract_frames.py --dir ./videos --resize 1280

Controls (if --preview is used):
    'q' — Quit early
    's' — Skip current frame (don't save)
"""

import cv2
import os
import argparse

SUPPORTED_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".wmv", ".flv", ".m4v"}


def extract_frames(video_path, output_dir="./frames", interval=0.5,
                   max_frames=None, resize_width=None, preview=False):
    """
    Extract frames from a video at regular time intervals.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        interval: Time between extracted frames in seconds
        max_frames: Maximum number of frames to extract (None = no limit)
        resize_width: Resize frames to this width (maintains aspect ratio)
        preview: Show each frame before saving
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    # Video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Duration: {duration:.1f}s ({total_frames} frames)")
    print(f"  Interval: {interval}s")

    expected_frames = int(duration / interval) if interval > 0 else total_frames
    if max_frames:
        expected_frames = min(expected_frames, max_frames)
    print(f"  Expected output: ~{expected_frames} frames")

    os.makedirs(output_dir, exist_ok=True)

    # Base name from video filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_interval = int(fps * interval) if fps > 0 else 1
    saved_count = 0
    frame_idx = 0

    print(f"\nExtracting to: {output_dir}/")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Resize if requested
            if resize_width and frame.shape[1] != resize_width:
                scale = resize_width / frame.shape[1]
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (resize_width, new_h))

            # Preview mode
            if preview:
                display = frame.copy()
                cv2.putText(display,
                            f"Frame {saved_count} | {frame_idx / fps:.1f}s | Press 's' to skip, 'q' to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Preview", display)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quit early by user")
                    break
                elif key == ord('s'):
                    frame_idx += 1
                    continue

            # Save frame
            timestamp = frame_idx / fps if fps > 0 else frame_idx
            filename = f"{base_name}_{saved_count:04d}_{timestamp:.1f}s.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

            # Progress
            if saved_count % 10 == 0:
                print(f"  Saved {saved_count} frames ({timestamp:.1f}s / {duration:.1f}s)")

            if max_frames and saved_count >= max_frames:
                print(f"[INFO] Reached max_frames ({max_frames})")
                break

        frame_idx += 1

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    print(f"\nDone! Saved {saved_count} frames to {output_dir}/")
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from all videos in a directory for YOLO training data"
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Path to directory containing video files"
    )
    parser.add_argument(
        "--output", type=str, default="./frames",
        help="Output directory for extracted frames (default: ./frames)"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Time between frames in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames to extract per video"
    )
    parser.add_argument(
        "--resize", type=int, default=None,
        help="Resize frames to this width (maintains aspect ratio)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Preview each frame before saving (press 's' to skip, 'q' to quit)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[ERROR] Directory not found: {args.dir}")
        return

    # Collect all supported video files
    video_files = sorted([
        os.path.join(args.dir, f)
        for f in os.listdir(args.dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if not video_files:
        print(f"[ERROR] No supported video files found in: {args.dir}")
        print(f"  Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    print(f"Found {len(video_files)} video(s) in {args.dir}")
    print("=" * 50)

    total_saved = 0
    for i, video_path in enumerate(video_files, start=1):
        print(f"\n[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
        saved = extract_frames(
            video_path=video_path,
            output_dir=args.output,
            interval=args.interval,
            max_frames=args.max_frames,
            resize_width=args.resize,
            preview=args.preview,
        )
        if saved is not None:
            total_saved += saved

    print("\n" + "=" * 50)
    print(f"All done! Saved {total_saved} total frames from {len(video_files)} video(s).")


if __name__ == "__main__":
    main()
