import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def threshold_image(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    _, binary_image = cv2.threshold(
        blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary_image


def count_objects(image):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 5
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    return len(contours)


def process_images(folder):
    images, filenames = load_images_from_folder(folder)
    results = []
    for img, filename in zip(images, filenames):
        gray = convert_to_grayscale(img)
        binary = threshold_image(gray)
        num_objects = count_objects(binary)
        results.append((filename, num_objects))
    return results


def main():
    folder = "bird_miniatures"
    results = process_images(folder)
    for filename, count in results:
        print(f"{filename}: {count} birds")

    if results:
        img = cv2.imread(os.path.join(folder, results[0][0]))
        gray = convert_to_grayscale(img)
        binary = threshold_image(gray)
        plt.imshow(binary, cmap="gray")
        plt.title(f"{results[0][0]}: {results[0][1]} birds")
        plt.show()


if __name__ == "__main__":
    main()

# E0454_TR0000_OB0010_T01_M02.jpg: 2 birds
# E0294_TR0000_OB0030_T01_M10.jpg: 1 birds
# E0089_TR0006_OB0366_T01_M16.jpg: 6 birds
# E0206_TR0001_OB0020_T01_M10.jpg: 6 birds
# E0418_TR0000_OB1797_T01_M04.jpg: 4 birds
# E0071_TR0001_OB0031_T01_M02.jpg: 5 birds
# E0453_TR0001_OB0301_T01_M02.jpg: 8 birds
# E0418_TR0000_OB1504_T01_M02.jpg: 2 birds
# E0453_TR0001_OB0564_T01_M02.jpg: 16 birds
# E0089_TR0005_OB2257_T01_M13.jpg: 6 birds
# E0098_TR0000_OB0098_T01_M04.jpg: 2 birds
# E0411_TR0001_OB0477_T01_M02.jpg: 2 birds
# E0222_TR0000_OB0077_T01_M16.jpg: 1 birds
# E0411_TR0001_OB0486_T01_M02.jpg: 2 birds
# E0418_TR0000_OB1594_T01_M04.jpg: 1 birds
# E0297_TR0000_OB0036_T01_M16.jpg: 1 birds
