"""Create a tiny synthetic visual dataset for the v1.0 learner."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='demo_visual_data')
    parser.add_argument('--count', type=int, default=20)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    root = Path(args.output)
    for label in ('circle', 'square', 'triangle'):
        (root / label).mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        for label in ('circle', 'square', 'triangle'):
            image = Image.new('RGB', (32, 32), (245, 245, 245))
            draw = ImageDraw.Draw(image)
            jitter = int(rng.integers(-3, 4))
            box = (8 + jitter, 8, 24 + jitter, 24)
            if label == 'circle': draw.ellipse(box, fill=(50, 100, 220))
            elif label == 'square': draw.rectangle(box, fill=(220, 80, 70))
            else: draw.polygon([(16 + jitter, 7), (7 + jitter, 24), (25 + jitter, 24)], fill=(70, 170, 90))
            image.save(root / label / f'{label}_{i:03d}.png')
    print(f'created {root.resolve()}')


if __name__ == '__main__':
    main()
