###########################IMPORTING MODULES####################3
import nemo.collections.asr as nemo_asr
import os
import numpy as np
import torch
import os
import subprocess
import locale
locale.getpreferredencoding = lambda: "UTF-8"
from pydub import AudioSegment
import sounddevice as sd
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer.IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from pydub.playback import play
import os
import re
import glob
import json
import tempfile
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import vits.commons as commons
import vits.utils as utils
import argparse
import subprocess
from vits.data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from vits.models import SynthesizerTrn
from scipy.io.wavfile import write
import google.generativeai as genai
import pyaudio
import wave
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import gradio as gr
from dotenv import load_dotenv
load_dotenv() 
#######################DEFINING CONSTANTS#####################################

GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")
prompt= os.getenv("gemini_prompt")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(os.getenv("LLM"))
BATCH_SIZE = 10
DEVICE = os.getenv("Default_device")
quantization = os.getenv("quantization")

LANG = "kan"
print(GOOGLE_API_KEY,prompt,model,BATCH_SIZE,DEVICE)

TOKEN_OFFSET = 100

FORMAT = pyaudio.paInt16
CHANNELS = int(os.getenv("CHANNELS_AUDIO"))
RATE = int(os.getenv("RATE"))
CHUNK = int(os.getenv("CHUNK"))
SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD"))  # Adjust this threshold based on your microphone sensitivity
SILENCE_DURATION = int(os.getenv('SILENCE_DURATION')) 
PRE_SOUND_DURATION = int(os.getenv("PRE_SOUND_DURATION"))  
########################FUNCTIONS#########################################


if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        yield

def load_model(model_path):
    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(model_path)
    asr_model.eval()
    asr_model.to(device='cuda')
    return asr_model

def transcribe(wav_file, asr_model,logprobs=False):
   
    if type(wav_file) != list:
        wav_file = [wav_file]
    
    with autocast():
        with torch.no_grad():
                return asr_model.transcribe(wav_file)#, logprobs=logprobs)
        
def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations

def download(lang, tgt_dir="./"):
  lang_fn, lang_dir = os.path.join(tgt_dir, lang+'.tar.gz'), os.path.join(tgt_dir, lang)
  cmd = ";".join([
        f"wget https://dl.fbaipublicfiles.com/mms/tts/{lang}.tar.gz -O {lang_fn}",
        f"tar zxvf {lang_fn}"
  ])
  print(f"Download model for language: {lang}")
  subprocess.check_output(cmd, shell=True)
  print(f"Model checkpoints in {lang_dir}: {os.listdir(lang_dir)}")
  return lang_dir
def preprocess_char(text, lang=None):
    """
    Special treatement of characters in certain languages
    """
    print(lang)
    if lang == 'ron':
        text = text.replace("ț", "ţ")
    return text

class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        '''
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, \
             tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = f"perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd +=  f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line =  re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text):
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        print(f"text after filtering OOV: {txt_filt}")
        return txt_filt

def preprocess_text(txt, text_mapper, hps, uroman_dir=None, lang=None):
    txt = preprocess_char(txt, lang=lang)
    is_uroman = hps.data.training_files.split('.')[-1] == 'uroman'
    if is_uroman:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if uroman_dir is None:
                cmd = f"git clone git@github.com:isi-nlp/uroman.git {tmp_dir}"
                print(cmd)
                subprocess.check_output(cmd, shell=True)
                uroman_dir = tmp_dir
            uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
            print(f"uromanize")
            txt = text_mapper.uromanize(txt, uroman_pl)
            print(f"uroman text: {txt}")
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    return txt
def numpy_to_audio_segment(np_array, frame_rate):
    audio_segment = AudioSegment(
        np_array.tobytes(), 
        frame_rate=frame_rate,
        sample_width=np_array.dtype.itemsize, 
        channels=1
    )
    return audio_segment
def generate_audio(txt,net_g,text_mapper,hps):
    print(f"text: {txt}")
    txt = preprocess_text(txt, text_mapper, hps, lang=LANG)
    stn_tst = text_mapper.get_text(txt, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(DEVICE)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(DEVICE)
        hyp = net_g.infer(
            x_tst, x_tst_lengths, noise_scale=.667,
            noise_scale_w=0.8, length_scale=1.0
        )[0][0,0].cpu().float().numpy()

    print(f"Generated audio") 
    # play(numpy_to_audio_segment(hyp, hps.data.sampling_rate))

    sd.play(hyp, hps.data.sampling_rate)
    sd.wait()
def is_silent(data, threshold):
    """Return 'True' if below the 'silent' threshold"""
    return np.mean(np.abs(np.frombuffer(data, dtype=np.int16))) < threshold

def wait_for_sound(stream):
    """Wait for sound to start"""
    print("Waiting for sound...")
    while True:
        data = stream.read(CHUNK)
        if not is_silent(data, SILENCE_THRESHOLD):
            print("Sound detected, start recording...")
            return [data]

def record_audio():
    """Record audio until silence is detected"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    
    frames = wait_for_sound(stream)
    silence_count = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        
        if is_silent(data, SILENCE_THRESHOLD):
            silence_count += 1
        else:
            silence_count = 0
        
        if silence_count > (RATE / CHUNK * SILENCE_DURATION):
            print("Silence detected, stopping recording.")
            break

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return frames

def save_audio(frames, filename):
    """Save the recorded audio to a WAV file"""
    # Convert frames to AudioSegment
    audio_data = b''.join(frames)
    audio_segment = AudioSegment(
        audio_data,
        frame_rate=RATE,
        sample_width=pyaudio.get_sample_size(FORMAT),
        channels=CHANNELS
    )
    
    # Detect non-silent parts
    non_silent_ranges = detect_nonsilent(audio_segment, min_silence_len=1000, silence_thresh=-40)
    
    if non_silent_ranges:
        # Extract non-silent parts
        start_trim = non_silent_ranges[0][0]
        end_trim = non_silent_ranges[-1][1]
        trimmed_audio = audio_segment[start_trim:end_trim]
    else:
        trimmed_audio = audio_segment

    # Save the trimmed audio
    trimmed_audio.export(filename, format="wav")
    print(f"Audio saved as {filename}")
 ###############MODEL LOADING##############

def load_all_models(lang_opt):
    if lang_opt=="1":

        ckpt_dir = "kan"
        path='Models/Kannada.nemo'
    elif lang_opt=="2":
        ckpt_dir="hin"
        path="Models/Hindi.nemo"
    elif lang_opt=="3":
        ckpt_dir="eng"
        path="Models/English.nemo"
        asr_model_kan=load_model(path)
        vocab_file = f"{ckpt_dir}/vocab.txt"
        config_file = f"{ckpt_dir}/config.json"
        assert os.path.isfile(config_file), f"{config_file} doesn't exist"
        hps = utils.get_hparams_from_file(config_file)
        text_mapper = TextMapper(vocab_file)
        net_g = SynthesizerTrn(
            len(text_mapper.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        net_g.to(DEVICE)
        _ = net_g.eval()

        g_pth = f"{ckpt_dir}/G_100000.pth"
        print(f"load {g_pth}")

        _ = utils.load_checkpoint(g_pth, net_g, None)
        return asr_model_kan,[net_g,text_mapper,hps]

    asr_model_kan=load_model(path)
    indic_en_ckpt_dir =  "ai4bharat/indictrans2-indic-en-dist-200M" # "ai4bharat/indictrans2-indic-en-1B" 
    indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir, "indic-en", quantization=quantization)
    ip = IndicProcessor(inference=True)
    en_indic_ckpt_dir =   "ai4bharat/indictrans2-en-indic-dist-200M"
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", quantization)
    vocab_file = f"{ckpt_dir}/vocab.txt"
    config_file = f"{ckpt_dir}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    net_g.to(DEVICE)
    _ = net_g.eval()

    g_pth = f"{ckpt_dir}/G_100000.pth"
    print(f"load {g_pth}")

    _ = utils.load_checkpoint(g_pth, net_g, None)
    
    return asr_model_kan,[indic_en_model,indic_en_tokenizer,ip],[en_indic_model,en_indic_tokenizer,ip],[net_g,text_mapper,hps]


##################MAIN DOING #################

def main(lang):
# lang=input("Enter the language\n 1, Kannada\n 2. Hindi\n 3. English\n\n Your input:  ")
    if lang=="1":
        asr_model_kan,lis_indic_en,lis_en_indic,lis_tts=load_all_models(lang)
        indic_en_model,indic_en_tokenizer,ip=lis_indic_en
        en_indic_model,en_indic_tokenizer,ip=lis_en_indic
        net_g,text_mapper,hps=lis_tts
        src_lang,tgt_lang="kan_Knda","eng_Latn"
    elif lang=="2":
        asr_model_kan,lis_indic_en,lis_en_indic,lis_tts=load_all_models(lang)
        indic_en_model,indic_en_tokenizer,ip=lis_indic_en
        en_indic_model,en_indic_tokenizer,ip=lis_en_indic
        net_g,text_mapper,hps=lis_tts
        src_lang,tgt_lang="hin_Deva","eng_Latn"
    elif lang=="3":
        asr_model_kan,lis_tts=load_all_models(lang)
        net_g,text_mapper,hps=lis_tts
    else:
        print("Invalid choice!!!!!")
        exit(0)
    gemini_history=[]
    while True:
        
            inp=''
            out=''
            frames = record_audio()
            if frames:
                save_audio(frames, "output.wav")
                path="output.wav"
                if lang in ["1","2"]:
                    inp=str(batch_translate(transcribe(path,asr_model_kan),src_lang,tgt_lang,indic_en_model, indic_en_tokenizer, ip)[0])
                else:
                    inp=str(transcribe(path,asr_model_kan)[0])
                if len(inp)==0:
                    continue
                inp1=inp
                inp=prompt+" "+inp #
                inp="".join(gemini_history)+inp
                
                gemini_history.append("User: "+inp1+"\n")
                print(inp)
                os.remove("output.wav")
                out=str(model.generate_content(inp).text).replace("**","").replace('*',"").replace(".","").split(".")
                gemini_history.append("Output: "+out[0]+"\n")
                print(gemini_history)
                if lang in ["1","2"]:
                    out=batch_translate(out,tgt_lang,src_lang,en_indic_model,en_indic_tokenizer,ip)
                # gemini_history.append("Output: "+out[0]+"\n")
            
                generate_audio(out[0],net_g,text_mapper,hps)
        
if __name__=="__main__":
    main()