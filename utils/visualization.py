import numpy as np
import PIL
from PIL import Image


def save_concat_imgs(images, img_names, save_path):
    '''
        Saves concatenated images to file
        Input:
        - images - list of np arrays of images (of length N each), which names are decribed in img_names
        - img_names - list if image names ('trg', 'ref', 'disp', 'depth', 'ref_inv_warp', 'ref_warped', 'trg-ref_warped')
        - filenames - list of N filenames
    '''
    imgs = [Image.fromarray(np.array(img_np*255).astype('uint8')) for img_np in images]

    # pick the image which is the smallest, and resize the others to match it
    min_shape = sorted([(np.sum(img.size), img.size) for img in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(img.resize(min_shape)) for img in imgs))

    # save combined image
    imgs_concat = PIL.Image.fromarray(imgs_comb)
    imgs_concat.save(save_path)





























