import pywt


def wt_filter(data, threshold):
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

    result = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

    return result
