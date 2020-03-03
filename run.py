import cv2
import pydicom
import numpy as np
from matplotlib import pyplot as plt


def load_dcm(filename):
    return pydicom.dcmread(f'data/{filename}')


def compute_snr(signal_power, noise_power):
    return np.sqrt(signal_power/noise_power)


def compute_cnr(signal_contrast, noise_power):
    return signal_contrast/np.sqrt(noise_power)


def main():
    filenames = ['PMD8540804318002412548_s04_T1_REST_Frame_1__PCARDM1.dcm',
                 'PMD1907987506279511791_s08_T1_STRESS02_Frame_1__PCARDM1.dcm']
    dcms = [load_dcm(f) for f in filenames]

    dcm_rest = dcms[0]
    img_rest = dcm_rest.pixel_array

    [print(f'{k}: {v}') for k, v in dcm_rest.items()]

    histogram = cv2.calcHist([img_rest.astype('float32')], [0], mask=None, histSize=[256], ranges=[0, 2**16])
    plt.subplot(121), plt.imshow(img_rest, cmap=plt.cm.bone)
    plt.subplot(122), plt.plot(histogram), plt.xticks(np.arange(0, 2**8+1, step=2**6), np.arange(0, 2**16+1, step=2**14), rotation=45)
    plt.show()

    # Convertir imagen a float para evitar errores por overflow.
    img_rest = img_rest.astype('float64')

    # Medidas de calidad de imagen
    noise_threshold = 300   # Medido en [T1]
    noise_mask = (img_rest < noise_threshold) * (img_rest > 0)
    noise_power = np.average(np.square(img_rest[noise_mask]))

    signal_mask = img_rest > noise_threshold
    signal_power = np.average(np.square(img_rest[signal_mask]))
    signal_contrast = np.max(img_rest[signal_mask] - np.min(img_rest[signal_mask]))

    assert( np.all(img_rest[noise_mask] > 0))

    print(f'# Medidas de calidad de la imagen')
    print(f'Potencia de la señal: {signal_power} [T1^2].')
    print(f'Contraste de la señal: {signal_contrast} [T1].')
    print(f'Potencia del ruido: {noise_power} [T1^2].')
    print(f'SNR: {compute_snr(signal_power, noise_power)} [1/1].')
    print(f'CNR: {compute_cnr(signal_contrast, noise_power)} [1/1].')

    plt.subplot(221), plt.imshow(signal_mask)
    plt.subplot(222), plt.imshow(noise_mask)
    plt.subplot(223), plt.imshow(img_rest * signal_mask, cmap=plt.cm.bone)
    plt.subplot(224), plt.imshow(img_rest * noise_mask, cmap=plt.cm.bone)
    plt.show()

    # Movimiento

    movement_thr = 3e3
    movement_img = dcms[0].pixel_array - dcms[1].pixel_array
    movement_mask = np.abs(movement_img) > movement_thr

    plt.subplot(221), plt.imshow(dcms[0].pixel_array, cmap=plt.cm.bone)
    plt.subplot(222), plt.imshow(dcms[1].pixel_array, cmap=plt.cm.bone)
    plt.subplot(223), plt.imshow(movement_img, cmap=plt.cm.bone)
    plt.subplot(224), plt.imshow(movement_mask)
    plt.show()



if __name__ == '__main__':
    main()
