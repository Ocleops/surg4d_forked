import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

BASE = "/home/tumai/team1/Ken/4DLangSplatSurgery/data/cholecseg8k/video01/language_features_default"

if __name__ == "__main__":
    # Load _f file as np array
    f_path = os.path.join(BASE, "frame_14939_endo_f.npy")
    f = np.load(f_path) # Shape: (#masks, 512)
    print(f"shape of f: {f.shape}")

    # Compute a 3-dim PCA over the #masks observations
    pca = PCA(n_components=3)
    pca.fit(f)
    f_pca = pca.transform(f) # Shape: (#masks, 3)
    print(f"shape of f_pca: {f_pca.shape}")
    min_f_pca = f_pca.min()
    max_f_pca = f_pca.max()
    print(f"min_f_pca: {min_f_pca}")
    print(f"max_f_pca: {max_f_pca}")
    # Shift and scale f_pca to be in the range [0, 255]
    f_pca = (f_pca - min_f_pca) / (max_f_pca - min_f_pca) * 255
    print(f"shape of f_pca after scaling: {f_pca.shape}")
    print(f"min_f_pca after scaling: {f_pca.min()}")
    print(f"max_f_pca after scaling: {f_pca.max()}")

    # Load s_file as np array
    s_path = os.path.join(BASE, "frame_14939_endo_s.npy")
    s = np.load(s_path)
    # Get rid of first dimension of s
    s = s.squeeze(0)
    print(f"shape of s: {s.shape}")

    # Iterate over all pixels in the s_file to create a new RGB image of shape (H, W, 3)
    rgb_image = np.zeros((s.shape[0], s.shape[1], 3))
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            # Use the value at the current pixel to retrieve the f_value
            segment_index = int(s[i, j])
            # print(f"segment_index: {segment_index}")
            # print(f"type of segment_index: {type(segment_index)}")
            if segment_index == -1:
                f_value = np.zeros(3)
            else:
                f_value = f_pca[segment_index]
            # print(f"f_value: {f_value}")
            rgb_image[i, j, :] = f_value

    # Save the RGB image in the same directory as the s_file
    rgb_image_path = os.path.join(BASE, "frame_14939_endo_rgb.png")
    Image.fromarray(rgb_image.astype(np.uint8), mode="RGB").save(rgb_image_path)

