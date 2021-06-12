import skimage.morphology as morph
from skimage import io, color
import matplotlib.pyplot as plt
from skimage import exposure
from skimage import filters
import numpy as np

# Save segmented image
def saveSegmentedImg(img):
    plt.figure(figsize = (10, 10))
    plt.axis("off")
    plt.imshow(img, cmap = "gray")
    plt.savefig('g.png', bbox_inches='tight', transparent=True, pad_inches=0)

# Convert to Grayscale using Luminance method
def convertLuminance(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    imgGray = 0.299 * R + 0.587 * G + 0.114 * B

    return imgGray.astype(np.uint8)

def scalingImage(img, minVal, maxVal):
    imax = np.max(img)
    imin = np.min(img)

    std = (img - imin) / (imax - imin)

    imgScaled = std * (maxVal - minVal) + minVal

    return imgScaled

# Histogram Equalization
def histogramEq(A):
    # Inicializamos o histograma
    hist = np.zeros(256).astype(int)

    # Verificamos cada tom para indicar a frequência na imagem
    for i in range(256):
        pixels_value_i = np.sum(A == i)
        hist[i] = pixels_value_i

    # Inicializamos para o histograma cumulativo
    histC = np.zeros(256).astype(int)

    # Intensidade 0
    histC[0] = hist[0]

    # Da intensidade 1 até 256
    for i in range(1,  256):
        histC[i] = hist[i] + histC[i-1]

    # Dimensões da imagem input
    N, M = A.shape
    
    # Inicializa espaço para guardar imagem equalizada
    A_eq = np.zeros([N,M]).astype(np.uint8)
    
    # Para cada valor de intensidade, transforma em uma nova intensidade
    for z in range(256):
        # Função de transformação
        s = ((256 - 1)/float(M * N)) * histC[z]
        
        # Para cada coordenada da imagem que tiver o valor z, atribui o valor equalizado
        A_eq[np.where(A == z)] = s
        
    return A_eq

def gaussianFilter(img, k = 15, sigma = 10):
    # Pega inputs para o filtro 2D
    n = k

    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
    filter2d = filt / np.sum(filt)

    # Centro para a subimagem
    center = n // 2

    # Padding com os números "refletidos" na borda considerando o tamanho do filtro
    padImg = np.pad(img, pad_width = center, mode = "symmetric")

    N, M = img.shape

    # Soma da multiplicação entre as matrizes, fazendo as convoluções devidas
    g = [[np.multiply(filter2d, padImg[(x - center):(x + center + 1), (y - center):(y + center + 1)]).sum() for y in range(center, M + center)] for x in range(center, N + center)]

    # Padronização da imagem
    g = scalingImage(g, 0, 1)

    return g


def thresholdRegion(img):
    gray_r = img.reshape(img.shape[0] * img.shape[1])

    for i in range(gray_r.shape[0]):
        if gray_r[i] > gray_r.mean():
            gray_r[i] = 3
        elif gray_r[i] > 0.5:
            gray_r[i] = 2
        elif gray_r[i] > 0.2:
            gray_r[i] = 1
        else:
            gray_r[i] = 0

    return gray_r.reshape(img.shape[0], img.shape[1])

def main():
    cellOriginal = io.imread('Data/mitosis/IMG_1681-1.jpg')

    cellGray = convertLuminance(cellOriginal)

    cellEq = histogramEq(cellGray)

    cellGauss = gaussianFilter(cellEq)

    cellThresh = thresholdRegion(cellGauss)

    saveSegmentedImg(cellThresh)

if __name__ == '__main__':
    main()