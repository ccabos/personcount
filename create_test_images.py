#!/usr/bin/env python3
"""
Creates synthetic test images for the person counter.
These images contain simple shapes representing people for testing purposes.
"""

import numpy as np
from PIL import Image, ImageDraw


def create_orchestra_image(output_path: str, num_rows: int = 3, persons_per_row: int = 8):
    """Creates a synthetic orchestra image with person-like shapes."""
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color=(40, 40, 50))  # Dark background
    draw = ImageDraw.Draw(img)

    # Draw stage floor
    draw.rectangle([0, height//2, width, height], fill=(80, 60, 40))

    # Draw person silhouettes (head + body)
    person_count = 0
    for row in range(num_rows):
        y_base = 200 + row * 120
        x_spacing = width // (persons_per_row + 1)

        for i in range(persons_per_row):
            x = x_spacing * (i + 1) + np.random.randint(-10, 10)

            # Head (circle)
            head_radius = 15 + row * 2
            head_y = y_base - 30
            draw.ellipse([x - head_radius, head_y - head_radius,
                         x + head_radius, head_y + head_radius],
                        fill=(200, 160, 140))

            # Body (rectangle)
            body_width = 30 + row * 5
            body_height = 60 + row * 10
            draw.rectangle([x - body_width//2, head_y + head_radius,
                           x + body_width//2, head_y + head_radius + body_height],
                          fill=(20, 20, 20))  # Black clothing

            person_count += 1

    img.save(output_path)
    print(f"Created {output_path} with {person_count} synthetic persons")
    return person_count


def create_audience_image(output_path: str, rows: int = 5, seats_per_row: int = 12):
    """Creates a synthetic audience/theater image."""
    width, height = 900, 700
    img = Image.new('RGB', (width, height), color=(30, 25, 35))
    draw = ImageDraw.Draw(img)

    person_count = 0
    for row in range(rows):
        y_base = 100 + row * 110
        x_spacing = width // (seats_per_row + 1)

        for i in range(seats_per_row):
            # Some seats might be empty
            if np.random.random() > 0.1:  # 90% occupancy
                x = x_spacing * (i + 1)

                # Head
                head_radius = 12
                draw.ellipse([x - head_radius, y_base - head_radius,
                             x + head_radius, y_base + head_radius],
                            fill=(np.random.randint(180, 220),
                                  np.random.randint(140, 180),
                                  np.random.randint(120, 160)))

                # Shoulders/upper body
                draw.rectangle([x - 20, y_base + head_radius,
                               x + 20, y_base + head_radius + 40],
                              fill=(np.random.randint(40, 100),
                                    np.random.randint(40, 100),
                                    np.random.randint(40, 100)))
                person_count += 1

        # Draw seat backs
        draw.rectangle([20, y_base + 60, width - 20, y_base + 75],
                      fill=(60, 40, 30))

    img.save(output_path)
    print(f"Created {output_path} with {person_count} synthetic persons")
    return person_count


def create_small_group_image(output_path: str, num_persons: int = 5):
    """Creates a simple image with a small group of people."""
    width, height = 400, 300
    img = Image.new('RGB', (width, height), color=(100, 150, 200))  # Sky blue
    draw = ImageDraw.Draw(img)

    # Ground
    draw.rectangle([0, height * 2//3, width, height], fill=(80, 120, 80))

    x_spacing = width // (num_persons + 1)
    for i in range(num_persons):
        x = x_spacing * (i + 1)
        y_base = height * 2 // 3

        # Head
        draw.ellipse([x - 15, y_base - 100, x + 15, y_base - 70],
                    fill=(210, 180, 160))

        # Body
        draw.rectangle([x - 20, y_base - 70, x + 20, y_base],
                      fill=(np.random.randint(50, 200),
                            np.random.randint(50, 200),
                            np.random.randint(50, 200)))

    img.save(output_path)
    print(f"Created {output_path} with {num_persons} synthetic persons")
    return num_persons


if __name__ == "__main__":
    import os

    output_dir = "test_images"
    os.makedirs(output_dir, exist_ok=True)

    create_orchestra_image(f"{output_dir}/orchestra_synthetic.jpg")
    create_audience_image(f"{output_dir}/audience_synthetic.jpg")
    create_small_group_image(f"{output_dir}/small_group.jpg", num_persons=5)

    print("\nTest images created successfully!")
