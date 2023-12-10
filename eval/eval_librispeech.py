import whisper
from whisper.normalizers import EnglishTextNormalizer

import os
import sys
import editdistance


target_dir=sys.argv[1]
model_tag = "large"
model = whisper.load_model(model_tag)
beam_size=5
options = whisper.DecodingOptions(
    task='transcribe',
    language='en',
    # Sampling-related
    temperature=0.0,
    sample_len=None,  # maximum number of tokens to sample
    best_of=None,  # number of independent sample trajectories, if t>0
    beam_size=beam_size,  # number of beams, if t==0
    patience=None,  # patience in beam search (arxiv:2204.05424)
    # "alpha" in Google NMT
    length_penalty=None,
    #
    prompt=None,
    prefix=None,
    # list of tokens to suppress
    suppress_tokens="-1",  # "-1" will suppress a set of symbols as defined in 'tokenizer.non_speech_tokens'
    suppress_blank=True,
    # time-stamp
    without_timestamps=True,
    max_initial_timestamp=1.0,
    # implementation details
    fp16=True,
)
normalizer = EnglishTextNormalizer()
logfilename="log_{target_dir}" \
    + f"_beam{beam_size}" if beam_size is not None and beam_size>1 else "" \
    + ".txt"
logfile = open(logfilename, "w")

def log_and_print(text):
    logfile.write(text + "\n")
    print(text)
def matching_text(audiopath):
    # get the text from the file path
    # e.g. "61-70968-0054.wav" -> "61-70968.trans.txt"
    basename = "/".join(audiopath.split("/")[:-1])
    filename = audiopath.split("/")[-1].split(".")[0]
    txtfilename = "-".join(filename.split("-")[:-1]) + '.trans.txt'
    # get the text from the transcription file
    # e.g. "61-70968.trans.txt" -> "THAT WAS THE TIME"
    with open(os.path.join(basename, txtfilename)) as f:
        for line in f:
            if line.startswith(filename):
                return line[len(filename)+1:].strip()
            
def decode_audio(audiopath):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audiopath)
    audio = whisper.pad_or_trim(audio)
    text = matching_text(audiopath)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(
        audio,
        n_mels=128 if model_tag == "large" else 80,
        ).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    lang = {max(probs, key=probs.get)}
    assert lang == {"en"}, f"Language not supported: {lang}"
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    result = whisper.decode(model, mel, options)
    # remove all characters except for english letters + space + apostrophe
    hyp = normalizer(result.text).split()
    ref = normalizer(text).split()

    # print the recognized text
    num_err = editdistance.eval(hyp, ref)
    num_words = len(hyp)
    log_and_print(f'REF: {ref}')
    log_and_print(f'HYP: {hyp}')
    log_and_print(f'#ERR/#WORDS = {num_err}/{num_words}')
    return {'hyp': hyp, 'ref': text, 'num_err': num_err, 'num_words': num_words}

if __name__ == "__main__":
    error_words = 0
    total_words = 0
    for root, dirs, files in os.walk(f"/home/sean/speechDB/librispeech/LibriSpeech/{target_dir}"):
        for name in files:
            if name.endswith(".wav"):
                result = decode_audio(os.path.join(root, name))
                error_words += result['num_err']
                total_words += result['num_words']
    log_and_print(f'===========================================')
    log_and_print(f'Total Error Words: {error_words}')
    log_and_print(f'Total Words: {total_words}')
    log_and_print(f'Total WER: {error_words/total_words}')