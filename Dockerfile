FROM nvcr.io/nvidia/nemo:23.10

RUN mkdir asr
WORKDIR asr

COPY . .

RUN wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/langid_ambernet/1.12.0/files?redirect=true&path=ambernet.nemo' -O ambernet.nemo
RUN wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/stt_en_conformer_ctc_large/1.10.0/files?redirect=true&path=stt_en_conformer_ctc_large.nemo' -O stt_en_conformer_ctc_large.nemo
RUN wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/stt_ru_conformer_ctc_large/1.13.0/files?redirect=true&path=stt_ru_conformer_ctc_large.nemo' -O stt_ru_conformer_ctc_large.nemo
RUN wget https://models.silero.ai/te_models/v2_4lang_q.pt -O v2_4lang_q.pt

RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt

EXPOSE 2600

CMD gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:app --preload --bind 0.0.0.0:2600 --timeout 60
