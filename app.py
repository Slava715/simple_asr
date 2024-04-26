from concurrent.futures import ProcessPoolExecutor
from pyctcdecode import build_ctcdecoder
import numpy as np
import math

import config
from extractor import NumberExtractor
number_extractor = NumberExtractor()

import torch
torch.set_num_threads(1)


def preproces_data(raw_file):
    data = np.array(np.frombuffer(raw_file, dtype=np.int16))
    
    return data/32768.0

def norm_ts(raw_ts):
    norm_ts = []
    
    for itm in raw_ts:
        norm_ts.append({"word": itm[0], "start": round(itm[1][0] * config.TIME_STEP, 3), "end": round(itm[1][1] * config.TIME_STEP, 3)})
    
    return norm_ts


beam_search_ru = build_ctcdecoder(
    labels = config.VOCAB_RU,
    #kenlm_model_path = "",
    #alpha = 0.5,
    #beta = 0.5
)

beam_search_en = build_ctcdecoder(
    labels = config.VOCAB_EN,
    #kenlm_model_path = "",
    #alpha = 0.5,
    #beta = 0.5
)


inited = False
initresult = None

def initwrapper(initfunc, initargs, f, x):
    global inited, initresult
    if not inited:
        inited = True
        initresult = initfunc(*initargs)
    return f(x)


def init_asr(model_path):
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.EncDecCTCModel.restore_from(model_path, map_location= config.DEVICE)
    model.eval()
    return model

def asr_audio(frame):
    length = torch.FloatTensor([frame.shape[0]])

    input_signal = torch.unsqueeze(frame, 0)
    
    log_probs, _, _ = initresult.forward(input_signal=input_signal.to( config.DEVICE), input_signal_length=length.to( config.DEVICE))
    
    return log_probs[0].detach().cpu().numpy()


def init_lang(model_path):
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path, map_location= config.DEVICE)
    model.eval()
    return model

def get_lang(frame):
    length = torch.FloatTensor([frame.shape[0]])
    
    input_signal = torch.unsqueeze(frame, 0)
    
    logits, _ = initresult.forward(input_signal=input_signal.to( config.DEVICE), input_signal_length=length.to( config.DEVICE))
    
    label_id = int(logits.argmax())
    
    return initresult._cfg['train_ds']['labels'][label_id]


def init_punct(model_path):
    return torch.package.PackageImporter(model_path).load_pickle("te_model", "model")

def get_punct(data):
    return initresult.enhance_text(data[0], lan=data[1])


pool_ru = ProcessPoolExecutor(config.NUM_WORKERS)
pool_en = ProcessPoolExecutor(config.NUM_WORKERS)
pool_lang = ProcessPoolExecutor(config.NUM_WORKERS)
pool_punct = ProcessPoolExecutor(config.NUM_WORKERS)


def asr_data(frame):
    lang = pool_lang.submit(initwrapper, init_lang, (config.MODEL_LANG,), get_lang, (frame)).result()
    
    if lang == "en":
        logits = pool_en.submit(initwrapper, init_asr, (config.MODEL_EN,), asr_audio, (frame)).result()
        transcript, _, raw_ts, _, _ = beam_search_en.decode_beams(logits)[0]
        
    else:
        logits = pool_ru.submit(initwrapper, init_asr, (config.MODEL_RU,), asr_audio, (frame)).result()
        transcript, _, raw_ts, _, _ = beam_search_ru.decode_beams(logits)[0]
        
    text = pool_punct.submit(initwrapper, init_punct, (config.MODEL_PUNCT,), get_punct, (transcript, lang)).result()
    
    if lang == "ru":
        text = number_extractor.replace_num(text, apply_regrouping=True)
    
    return text, norm_ts(raw_ts)
    
    
import uvicorn
import json
from fastapi import FastAPI, Request, Response

app = FastAPI()

@app.route('/asr_file', methods=['POST'])
async def asr_file_raw(request: Request):
    try:
        raw_file = await request.body()
        frame = torch.FloatTensor(preproces_data(raw_file))
        text, result = asr_data(frame)
        
    except Exception as e:
        print("!! Err : ", e)
        return Response(content=json.dumps({"response_code": "err", "error": e}), media_type="application/json")
    
    return Response(content=json.dumps({"response_code": "ok", "text": text, "result": result}), media_type="application/json")
