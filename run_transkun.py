import os
import argparse
from tqdm import tqdm


def removeExtension(file):
    return ".".join(os.path.basename(file).split('.')[:-1])


def transcribe(file, outfolder):
    print(file)

    path = os.path.join(outfolder, removeExtension(file)+".mid")
    os.system('transkun "%s" "%s" --device cuda'%(file,path))
    return path


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    for score_name in tqdm(os.listdir(input_dir)):
        score_dir = os.path.join(input_dir, score_name)
        out_score_dir = os.path.join(output_dir, score_name)
        os.makedirs(out_score_dir, exist_ok=True)
        for audio_name in os.listdir(score_dir):
            if not audio_name.endswith('.mp3'):
                continue
            audio_path = os.path.join(score_dir, audio_name)
            transcribe(audio_path, out_score_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='path to audio directory')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output MIDI directory')   
    
    args = parser.parse_args()

    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'

    main(args)
