from sys import argv
from argparse import ArgumentParser
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def convert_video(input_file, output_file):
    clip = VideoFileClip(input_file)
    clip.write_videofile(output_file, codec='libx264')

# def convert_to_mp4(mkv_file):
#     no_extension = str(os.path.splitext(mkv_file))
#     with_mp4 = no_extension + ".mp4"
#     ffmpeg.input(mkv_file).output(with_mp4).run()
#     print("Finished converting {}".format(no_extension))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()

    cam_list = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    # light_list= ['L030','L069','L058','L063','L250','L213','L013','L313']
    # cameras = os.listdir(args.input_dir)
    # cameras = []
    # for c in cam_list:
    #     for l in light_list:
    #         cameras.append(f'{c}_{l}')
            
    for cam in tqdm(cam_list):
        if cam == 'preprocess':
            continue
        inp = os.path.join(args.input_dir,f'{cam}/rgb_camspc_rot.mp40001-0100.mkv')
        out = os.path.join(args.output_dir,f'{cam}.mp4')
        convert_video(inp,out)

    


