from torch.multiprocessing import Process, Queue, Array, Semaphore
from pyctcdecode import build_ctcdecoder
from time import sleep
import numpy as np
import torch
import math

import config
from extractor import NumberExtractor
number_extractor = NumberExtractor()


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


class ProcessWraper:
    def __init__(self, process_function, args, workers=config.NUM_WORKERS, delay=config.DELAY_WORKER):        
        self.manage_lock = Semaphore()
        
        self.delay = delay
        self.workers = workers
        self.worker_in = []
        self.worker_out = []
        self.worker_state = Array('i', range(config.NUM_WORKERS))
        self.worker_process = Array('i', range(config.NUM_WORKERS))

        for n in range(config.NUM_WORKERS):
            self.worker_in.append(Queue())
            self.worker_out.append(Queue())
            self.worker_state[n] = 0

            p = Process(target=process_function, args=(self.worker_in[n], self.worker_out[n], delay, args,))
            p.start()
            self.worker_process[n] = p.pid

    def manage(self, num_worker=None):
        self.manage_lock.acquire()
        
        try:
            if num_worker == None:
                for n in range(self.workers):
                    if self.worker_state[n] == 0:
                        self.worker_state[n] = 1
                        num_worker = n
                        break
            else:
                if self.worker_state[num_worker] == 1:
                    self.worker_state[num_worker] = 0
                    num_worker = None

        finally:
            self.manage_lock.release()
            
        return num_worker
    
    def run(self, data):
        cnt = 0
        num_worker = None
        while cnt < 600 / self.delay:
            num_worker = self.manage(None)
            if num_worker != None:
                break
                
            cnt = cnt + 1
            sleep(self.delay)
            
        if num_worker == None:
            print("error get worker")
        
        self.worker_in[num_worker].put(data)
        cnt = 0
        result = None
        empty = True
        while cnt < 600 / self.delay:   
            if self.worker_out[num_worker].empty() == False:
                result = self.worker_out[num_worker].get()
                empty = False
                break
            cnt = cnt + 1
            if cnt * self.delay / 1.0 > 0 and cnt * self.delay % 1.0 == 0:
                print("trying to get a worker again sec: " + str(cnt * self.delay / 1.0))
            sleep(self.delay)
            
        if self.manage(num_worker) != None:
            print("error release worker: " + str(num_worker))
            
        return result
    
    
def asr_process(queue_in, queue_out, delay, model_path):   
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.EncDecCTCModel.restore_from(model_path, map_location = config.DEVICE)
    model.eval()

    while True:       
        sleep(delay)

        if queue_in.empty() == False:

            file_np = queue_in.get()

            length = torch.FloatTensor([file_np.shape[0]])

            input_signal = torch.unsqueeze(file_np, 0)

            log_probs, _, _ = model.forward(input_signal=input_signal.to(config.DEVICE), input_signal_length=length.to(config.DEVICE))

            queue_out.put(log_probs[0].detach().cpu().numpy())
            
            
def lang_process(queue_in, queue_out, delay, model_path):   
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path, map_location= config.DEVICE)
    model.eval()

    while True:       
        sleep(delay)

        if queue_in.empty() == False:

            frame = queue_in.get()

            length = torch.FloatTensor([frame.shape[0]])

            input_signal = torch.unsqueeze(frame, 0)

            logits, _ = model.forward(input_signal=input_signal.to( config.DEVICE), input_signal_length=length.to( config.DEVICE))

            label_id = int(logits.argmax())

            queue_out.put(model._cfg['train_ds']['labels'][label_id])
            

def punct_process(queue_in, queue_out, delay, model_path):   
    model = torch.package.PackageImporter(model_path).load_pickle("te_model", "model")

    while True:       
        sleep(delay)

        if queue_in.empty() == False:

            data = queue_in.get()
            
            queue_out.put(model.enhance_text(data[0], lan=data[1]))


pool_ru = ProcessWraper(asr_process, config.MODEL_RU, config.NUM_WORKERS, config.DELAY_WORKER)
pool_en = ProcessWraper(asr_process, config.MODEL_EN, config.NUM_WORKERS, config.DELAY_WORKER)
pool_lang = ProcessWraper(lang_process, config.MODEL_LANG, config.NUM_WORKERS, config.DELAY_WORKER)
pool_punct = ProcessWraper(punct_process, config.MODEL_PUNCT, config.NUM_WORKERS, config.DELAY_WORKER)


def asr_data(frame):
    global pool_ru, pool_en, pool_lang, pool_punct
    
    lang = pool_lang.run(frame)
    
    if lang == "en":
        logits = pool_en.run(frame)
        transcript, _, raw_ts, _, _ = beam_search_en.decode_beams(logits)[0]
        
    else:
        logits = pool_ru.run(frame)
        transcript, _, raw_ts, _, _ = beam_search_ru.decode_beams(logits)[0]
        
    text = pool_punct.run((transcript, lang))
    
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
