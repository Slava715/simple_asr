import soundfile as sf
import numpy as np
import librosa
import argparse

def change_speed(x, factor):
    if factor == 1:
        return x

    nfft = 256
    hop = 64

    stft = librosa.core.stft(x, n_fft=nfft).transpose()
    stft_cols = stft.shape[1]

    times = np.arange(0, stft.shape[0], factor)
    phase_adv = (2 * np.pi * hop * np.arange(0, stft_cols))/ nfft
    stft = np.concatenate((stft, np.zeros((1, stft_cols))), axis=0)

    indices = np.floor(times).astype(np.int)
    alpha = np.expand_dims(times - np.floor(times), axis=1)
    mag = (1. - alpha) * np.absolute(stft[indices, :]) + alpha * np.absolute(stft[indices + 1, :])

    dphi = np.angle(stft[indices + 1, :]) - np.angle(stft[indices, :]) - phase_adv
    dphi = dphi - 2 * np.pi * np.floor(dphi/(2 * np.pi))

    phase_adv_acc = np.matmul(np.expand_dims(np.arange(len(times) + 1),axis=1), np.expand_dims(phase_adv, axis=0))
    phase = np.concatenate( (np.zeros((1, stft_cols)), np.cumsum(dphi, axis=0)), axis=0) + phase_adv_acc
    phase += np.angle(stft[0, :])

    stft_new = mag * np.exp(phase[:-1, :] * 1j)

    return librosa.core.istft(stft_new.transpose())

parser = argparse.ArgumentParser(description='Модификация аудиофайла.')
parser.add_argument('--in_file', required=True, help='исходный файл')
parser.add_argument('--out_file', help='изменённый файл')
parser.add_argument('--volume', help='громкость')
parser.add_argument('--speed', help='скорость')

args = parser.parse_args()

signal, sr = librosa.load(args.in_file, sr = None)

if args.volume != None:
    signal = signal * float(args.volume)
    
if args.speed != None:
    signal = change_speed(signal, float(args.speed))

if args.out_file == None:  
    args.out_file = 'change_' + args.in_file

sf.write(args.out_file, signal, sr)

print("Готово!", args.out_file)
