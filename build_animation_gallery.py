"""Build a scrollable HTML gallery for generated telemetry animations."""

from __future__ import annotations

import argparse
import html
import os


DEFAULT_ANIMATIONS = [
    (
        "Longitudinal Acceleration",
        "afternoon_accel_animation.gif",
        "Track color shows the smoothed forward/backward acceleration channel.",
    ),
    (
        "Speed",
        "afternoon_speed_animation.gif",
        "Track color shows GPX-derived speed after the scale fix.",
    ),
    (
        "Current",
        "afternoon_current_animation.gif",
        "Track color shows current draw from the telemetry dump.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a scrollable HTML page for telemetry animation GIFs."
    )
    parser.add_argument(
        "--output",
        default="outputs/afternoon_animation_gallery.html",
        help="HTML gallery path.",
    )
    parser.add_argument(
        "--title",
        default="UTSM Afternoon Telemetry Animation Gallery",
    )
    return parser.parse_args()


def build_gallery(output_path: str, title: str) -> None:
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    sections = []
    for section_title, filename, description in DEFAULT_ANIMATIONS:
        image_path = filename
        full_path = os.path.join(out_dir, filename)
        if not os.path.exists(full_path):
            print(f"WARNING: missing animation: {full_path}")
        sections.append(
            f"""
            <section class="animation-card">
              <div class="copy">
                <h2>{html.escape(section_title)}</h2>
                <p>{html.escape(description)}</p>
              </div>
              <img src="{html.escape(image_path)}" alt="{html.escape(section_title)} animation">
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f5f7;
      color: #161a1d;
    }}
    body {{
      margin: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255, 255, 255, 0.94);
      border-bottom: 1px solid #d7dbe0;
      padding: 16px 24px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      font-weight: 700;
    }}
    header p {{
      margin: 0;
      color: #4a535c;
      font-size: 14px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    .animation-card {{
      background: #ffffff;
      border: 1px solid #d7dbe0;
      border-radius: 8px;
      margin-bottom: 28px;
      padding: 18px;
    }}
    .copy {{
      margin-bottom: 12px;
    }}
    h2 {{
      margin: 0 0 6px;
      font-size: 18px;
    }}
    .copy p {{
      margin: 0;
      color: #4a535c;
      font-size: 14px;
    }}
    img {{
      display: block;
      width: 100%;
      max-width: 1040px;
      height: auto;
      margin: 0 auto;
      border: 1px solid #e1e4e8;
      background: #ffffff;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Scroll down to compare the same afternoon run colored by acceleration, speed, and current.</p>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(document)
    print(f"Wrote gallery: {output_path}")


def main() -> int:
    args = parse_args()
    build_gallery(args.output, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
