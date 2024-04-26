# simple_asr

Simple multi language asr based on Nvidia nemo framework and Siero punctuations models

git clone https://github.com/Slava715/simple_asr.git

cd simple_asr

sudo docker build -t simple_asr .

sudo docker run simple_asr -p 2600:2600

There are also several tests

python3 test_1.py --in_file test_en.wav --speed 1.5

python3 test_1.py --in_file test_ru.wav --volume 1.5

python3 test_2.py --in_file test_en.wav --asr_url http://localhost:2600/asr_file

python3 test_2.py --in_file test_ru.wav --asr_url http://localhost:2600/asr_file
