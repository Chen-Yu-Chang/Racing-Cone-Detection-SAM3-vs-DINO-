"""
extract_frames.py

Pulls camera frames out of a single .mcap file so we have a handful of test
images to run Grounding DINO and SAM3 on.

IMPORTANT: check with your team first whether there's already an internal
util for this (PerceptionsLibrary / driverless-ml-dev likely has one, since
every perception member needs to pull frames out of mcap/rosbag data). If
one exists, use it instead - it'll already know your exact topic names and
message types. This script is a generic fallback in case there isn't one
handy, or you just want something standalone for this quick test.

Usage:
    python extract_frames.py --mcap path/to/session.mcap --list-topics
    python extract_frames.py --mcap path/to/session.mcap --topic /camera/image_raw --every-n 30 --max-frames 15 --out frames/

Step 1: always run with --list-topics first. MCAP files can contain many
topics (LiDAR, camera, IMU, CAN...) - you need to see the exact topic name
and message type for your camera before extracting.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader


def list_topics(mcap_path: str):
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        print(f"\nTopics found in {mcap_path}:\n")
        for channel_id, channel in summary.channels.items():
            schema = summary.schemas.get(channel.schema_id)
            schema_name = schema.name if schema else "?"
            print(f"  {channel.topic:40s}  msgtype={schema_name}")
        print()


def decode_image_message(schema_name: str, msg_bytes: bytes, typestore):
    """
    Decodes a single ROS2 message's raw CDR bytes into a numpy BGR image.
    Handles both sensor_msgs/msg/Image (raw) and sensor_msgs/msg/CompressedImage
    (jpeg/png bytes), which cover the vast majority of ROS2 camera topics.
    """
    msg = typestore.deserialize_cdr(msg_bytes, schema_name)

    if schema_name.endswith("CompressedImage"):
        # msg.data is already jpeg/png-encoded bytes
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    if schema_name.endswith("Image"):
        # raw encoding - handle the common ones; extend if yours differs
        h, w = msg.height, msg.width
        encoding = msg.encoding.lower()
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        if encoding in ("rgb8", "bgr8"):
            img = buf.reshape(h, w, 3)
            if encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        if encoding in ("mono8",):
            img = buf.reshape(h, w)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        raise ValueError(
            f"Unhandled raw image encoding '{msg.encoding}' - add a case for it "
            f"in decode_image_message(), or check if your camera actually "
            f"publishes CompressedImage instead."
        )

    raise ValueError(f"'{schema_name}' doesn't look like an image message type")


def extract(mcap_path: str, topic: str, every_n: int, max_frames: int, out_dir: str):
    from rosbags.typesys import get_typestore, Stores

    typestore = get_typestore(Stores.ROS2_HUMBLE)  # adjust if your team is on a
                                                     # different ROS2 distro's msg defs

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    count = 0
    saved = 0
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()

        # find schema name for the requested topic
        topic_schema = None
        for channel in summary.channels.values():
            if channel.topic == topic:
                schema = summary.schemas.get(channel.schema_id)
                topic_schema = schema.name
                break
        if topic_schema is None:
            raise SystemExit(
                f"Topic '{topic}' not found. Run with --list-topics to see what's available."
            )

        for schema, channel, message in reader.iter_messages(topics=[topic]):
            if count % every_n == 0:
                img = decode_image_message(topic_schema, message.data, typestore)
                if img is not None:
                    out_path = os.path.join(out_dir, f"frame_{saved:04d}.jpg")
                    cv2.imwrite(out_path, img)
                    saved += 1
                    print(f"saved {out_path}  (t={message.log_time})")
                if saved >= max_frames:
                    break
            count += 1

    print(f"\nDone. Extracted {saved} frames from {count} messages on '{topic}' -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True, help="path to the .mcap file")
    ap.add_argument("--list-topics", action="store_true")
    ap.add_argument("--topic", help="camera topic to pull frames from")
    ap.add_argument("--every-n", type=int, default=30,
                     help="save 1 out of every N camera messages (default 30, "
                          "so on a ~20-30fps feed you get roughly 1 frame/sec)")
    ap.add_argument("--max-frames", type=int, default=15,
                     help="stop after this many saved frames - keep this small "
                          "for a quick DINO-vs-SAM3 eyeball test")
    ap.add_argument("--out", default="frames")
    args = ap.parse_args()

    if args.list_topics:
        list_topics(args.mcap)
    else:
        if not args.topic:
            raise SystemExit("--topic is required unless you're using --list-topics")
        extract(args.mcap, args.topic, args.every_n, args.max_frames, args.out)
