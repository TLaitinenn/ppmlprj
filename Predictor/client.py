
from encryption_config import config
from encryption_handler.encryption_handler import EncryptionHandler
import numpy as np
import time
import sys
from matplotlib import image


def main():
    # Load test image and reshape to 1-D array
    imfile = "Predictor/sample_image.png"
    img = image.imread(imfile).reshape(-1)
    # Encrypt the image
    start1=time.clock()
    handler = EncryptionHandler(config)
    op = handler.package
    ln = img.shape[0]
    start = time.clock()
    encrypted_image = []
    print(type(img), img.dtype, ln, img.shape)
    for i in range(ln):
        encrypted_image.append((op.Ciphertext()))
        handler.encryptor.encrypt(handler.encoder.encode(img[i]), encrypted_image[i])
    print("time taken for encrypting image:  " + (str)(time.clock() - start)+"s")
    print("Noise budget in fresh encryption: " + (str)(handler.decryptor.invariant_noise_budget(encrypted_image[100])) + " bits")
    # Send encrypted image to predictor
if __name__ == '__main__':
    main()