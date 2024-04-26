from pydub import AudioSegment
import requests
import argparse
import json
import sys
import io

parser = argparse.ArgumentParser(description='Расшифровка аудио в текст.')
parser.add_argument('--in_file', required=True, help='аудио файл')
parser.add_argument('--log_file', default='log.json', help='лог файл')
parser.add_argument('--asr_url', default='http://0.0.0.0:2600/asr_file', help='адрес сервера')

args = parser.parse_args()

sound = AudioSegment.from_file(args.in_file).set_frame_rate(16000).set_sample_width(2)
channels = sound.split_to_mono()
f_wav = io.BytesIO()

with open(args.log_file, 'a') as out:
    for channel in channels:
        channel.export(f_wav, format="raw")
        response = requests.post(args.asr_url, data = f_wav)
        json.dump(response.json(), out)
        out.write('\n')
        print(response.json())
