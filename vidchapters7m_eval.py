import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as tqdm
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from vidchapters7m_dataloader import VidChapters7M_Dataset
    

def get_args_vidchap():
    # build args
    args = {
        "json_path": '/raid/1moritz/datasets/VidChapters-7M/chapters_dvc_test.json',
        "video_folder": '/raid/1moritz/datasets/VidChapters-7M/chapter_clips',
        "audio_folder": '/raid/1moritz/datasets/VidChapters-7M/chapter_audios',
        "asr_folder": '/raid/1moritz/datasets/VidChapters-7M/chapter_asr',
        "summary_folder": '/raid/1moritz/datasets/VidChapters-7M/chapter_summary_asr',
        "batch_size_val": 8,
        "num_thread_reader": 1,
        "cache_dir": '/raid/1moritz/models/languagebind/downloaded_weights',
    }
    args = EasyDict(args)
    return args

def run_eval(model: imagebind_model.ImageBindModel, dataloader: DataLoader, device: torch.device):
    batch_sentences_embeddings, batch_videos_embeddings, batch_audios_embeddings, batch_asr_embeddings = [], [], [], []
    batch_summaries_embeddings = []
    # Calculate embeddings
    for bid, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        sentences, video_paths, audio_paths, asr_texts, summaries = batch

        if not isinstance(sentences, list):
            sentences = list(sentences)
        if not isinstance(video_paths, list):
            video_paths= list(video_paths)
        if not isinstance(audio_paths, list):
            audio_paths = list(audio_paths)
        if not isinstance(asr_texts, list):
            asr_texts = list(asr_texts)
        if not isinstance(summaries, list):
            summaries = list(summaries)

        # Load data
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(sentences, device),
            ModalityType.VISION: data.load_and_transform_video_data(video_paths, device),
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
        }
        asr_inputs = {
            ModalityType.TEXT: data.load_and_transform_text(asr_texts, device),
        }
        summary_inputs = {
            ModalityType.TEXT: data.load_and_transform_text(summaries, device),
        }
        
        with torch.no_grad():
            embeddings = model(inputs)
            asr_embeddings = model(asr_inputs)
            summary_embeddings = model(summary_inputs)

        batch_sentences_embeddings.append(embeddings[ModalityType.TEXT])
        batch_videos_embeddings.append(embeddings[ModalityType.VISION])
        batch_audios_embeddings.append(embeddings[ModalityType.AUDIO])
        batch_asr_embeddings.append(asr_embeddings[ModalityType.TEXT])
        batch_summaries_embeddings.append(summary_embeddings[ModalityType.TEXT])

    # Create similarity matrix
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_videos_embeddings)

    # Log metrics Text-to-Video
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    print('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    print(f"VidChapters Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    print(f"VidChapters Video-to-Text:")
    print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
    
    # Log metrics Text-to-Audio
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_audios_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ta_metrics = compute_metrics(sim_matrix)
    at_metrics = compute_metrics(sim_matrix.T)
    print(f"VidChapters Text-to-Audio:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(ta_metrics['R1'], ta_metrics['R5'], ta_metrics['R10'], ta_metrics['MR'], ta_metrics['MeanR']))
    print(f"VidChapters Audio-to-Text:")
    print('\t>>>  A2T$R@1: {:.1f} - A2T$R@5: {:.1f} - A2T$R@10: {:.1f} - A2T$Median R: {:.1f} - A2T$Mean R: {:.1f}'.
                format(at_metrics['R1'], at_metrics['R5'], at_metrics['R10'], at_metrics['MR'], at_metrics['MeanR']))
    
    # Log metrics Audio-to-Video
    sim_matrix = create_sim_matrix(batch_audios_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    av_metrics = compute_metrics(sim_matrix)
    va_metrics = compute_metrics(sim_matrix.T)
    print(f"VidChapters Audio-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(av_metrics['R1'], av_metrics['R5'], av_metrics['R10'], av_metrics['MR'], av_metrics['MeanR']))
    print(f"VidChapters Video-to-Audio:")
    print('\t>>>  V2A$R@1: {:.1f} - V2A$R@5: {:.1f} - V2A$R@10: {:.1f} - V2A$Median R: {:.1f} - V2A$Mean R: {:.1f}'.
                format(va_metrics['R1'], va_metrics['R5'], va_metrics['R10'], va_metrics['MR'], va_metrics['MeanR']))
    
    # Log metrics Text-to-ASR
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_asr_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ta_metrics = compute_metrics(sim_matrix)
    at_metrics = compute_metrics(sim_matrix.T)
    print(f"VidChapters Text-to-ASR:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(ta_metrics['R1'], ta_metrics['R5'], ta_metrics['R10'], ta_metrics['MR'], ta_metrics['MeanR']))
    print(f"VidChapters ASR-to-Text:")
    print('\t>>>  Asr2T$R@1: {:.1f} - Asr2T$R@5: {:.1f} - Asr2T$R@10: {:.1f} - Asr2T$Median R: {:.1f} - Asr2T$Mean R: {:.1f}'.
                format(at_metrics['R1'], at_metrics['R5'], at_metrics['R10'], at_metrics['MR'], at_metrics['MeanR']))
    
    # Log metrics ASR-to-Video
    sim_matrix = create_sim_matrix(batch_asr_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    av_metrics = compute_metrics(sim_matrix)
    va_metrics = compute_metrics(sim_matrix.T)
    print(f"VidChapters ASR-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(av_metrics['R1'], av_metrics['R5'], av_metrics['R10'], av_metrics['MR'], av_metrics['MeanR']))
    print(f"VidChapters Video-to-ASR:")
    print('\t>>>  V2Asr$R@1: {:.1f} - V2Asr$R@5: {:.1f} - V2Asr$R@10: {:.1f} - V2Asr$Median R: {:.1f} - V2Asr$Mean R: {:.1f}'.
                format(va_metrics['R1'], va_metrics['R5'], va_metrics['R10'], va_metrics['MR'], va_metrics['MeanR']))
    
    # Log metrics Text-to-Summary
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_summaries_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ts_metrics = compute_metrics(sim_matrix)
    st_metrics = compute_metrics(sim_matrix.T)
    print(f"VidChapters Text-to-Summary_ASR:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(ts_metrics['R1'], ts_metrics['R5'], ts_metrics['R10'], ts_metrics['MR'], ts_metrics['MeanR']))
    print(f"VidChapters Summary_ASR-to-Text:")
    print('\t>>>  Sum2T$R@1: {:.1f} - Sum2T$R@5: {:.1f} - Sum2T$R@10: {:.1f} - Sum2T$Median R: {:.1f} - Sum2T$Mean R: {:.1f}'.
                format(st_metrics['R1'], st_metrics['R5'], st_metrics['R10'], st_metrics['MR'], st_metrics['MeanR']))
    
    # Log metrics Summary-to-Video
    sim_matrix = create_sim_matrix(batch_summaries_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    sv_metrics = compute_metrics(sim_matrix)
    vs_metrics = compute_metrics(sim_matrix.T)
    print(f"VidChapters Summary_ASR-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(sv_metrics['R1'], sv_metrics['R5'], sv_metrics['R10'], sv_metrics['MR'], sv_metrics['MeanR']))
    print(f"VidChapters Video-to-Summary_ASR:")
    print('\t>>>  V2Sum$R@1: {:.1f} - V2Sum$R@5: {:.1f} - V2Sum$R@10: {:.1f} - V2Sum$Median R: {:.1f} - V2Sum$Mean R: {:.1f}'.
                format(vs_metrics['R1'], vs_metrics['R5'], vs_metrics['R10'], vs_metrics['MR'], vs_metrics['MeanR']))

def create_sim_matrix(batch_sentences_embeddings, batch_videos_embeddings):
    """Calculate embedding vector product for similarity and download result to CPU
    
        Returns: 
            sim_matrix (Text X Video)
    """
    sim_matrix = []
    for idx1 in range(len(batch_sentences_embeddings)):
        sequence_output = batch_sentences_embeddings[idx1]
        each_row = []
        for idx2 in range(len(batch_videos_embeddings)):
            visual_output = batch_videos_embeddings[idx2]
            b1b2 =  sequence_output @ visual_output.T
            b1b2 = b1b2.cpu().detach().numpy()
            each_row.append(b1b2)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    return sim_matrix

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    # metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def main():
    assert torch.cuda.is_available()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    args = get_args_vidchap()

    dataloader_vidchap = DataLoader(
        VidChapters7M_Dataset(
            json_path=args.json_path, 
            video_folder=args.video_folder, 
            audio_folder=args.audio_folder,
            asr_folder=args.asr_folder,
            summary_folder=args.summary_folder),
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )

    return run_eval(model, dataloader_vidchap, device)


if __name__ == '__main__':
    main()
"""
VidChapters sim matrix size: 895, 895
	 Length-T: 895, Length-V:895
VidChapters Text-to-Video:
	>>>  R@1: 26.4 - R@5: 44.9 - R@10: 52.2 - Median R: 9.0 - Mean R: 111.5
VidChapters Video-to-Text:
	>>>  V2T$R@1: 9.6 - V2T$R@5: 17.5 - V2T$R@10: 21.1 - V2T$Median R: 58.0 - V2T$Mean R: 94.8
VidChapters sim matrix size: 895, 895
VidChapters Text-to-Audio:
	>>>  R@1: 0.4 - R@5: 3.4 - R@10: 6.1 - Median R: 239.0 - Mean R: 292.2
VidChapters Audio-to-Text:
	>>>  A2T$R@1: 0.5 - A2T$R@5: 1.7 - A2T$R@10: 2.9 - A2T$Median R: 174.0 - A2T$Mean R: 229.0
VidChapters sim matrix size: 895, 895
VidChapters Audio-to-Video:
	>>>  R@1: 4.9 - R@5: 18.0 - R@10: 29.9 - Median R: 31.0 - Mean R: 90.2
VidChapters Video-to-Audio:
	>>>  V2A$R@1: 4.1 - V2A$R@5: 19.2 - V2A$R@10: 30.1 - V2A$Median R: 33.0 - V2A$Mean R: 85.8
"""

"""
VidChapters sim matrix size: 895, 895
	 Length-T: 895, Length-V:895
VidChapters Text-to-Video:
	>>>  R@1: 26.4 - R@5: 44.9 - R@10: 52.2 - Median R: 9.0 - Mean R: 111.5
VidChapters Video-to-Text:
	>>>  V2T$R@1: 9.6 - V2T$R@5: 17.5 - V2T$R@10: 21.1 - V2T$Median R: 58.0 - V2T$Mean R: 94.8
VidChapters sim matrix size: 895, 895
VidChapters Text-to-Audio:
	>>>  R@1: 0.4 - R@5: 3.4 - R@10: 6.1 - Median R: 239.0 - Mean R: 292.2
VidChapters Audio-to-Text:
	>>>  A2T$R@1: 0.5 - A2T$R@5: 1.7 - A2T$R@10: 2.9 - A2T$Median R: 174.0 - A2T$Mean R: 229.0
VidChapters sim matrix size: 895, 895
VidChapters Audio-to-Video:
	>>>  R@1: 4.9 - R@5: 18.0 - R@10: 29.9 - Median R: 31.0 - Mean R: 90.2
VidChapters Video-to-Audio:
	>>>  V2A$R@1: 4.1 - V2A$R@5: 19.2 - V2A$R@10: 30.1 - V2A$Median R: 33.0 - V2A$Mean R: 85.8
VidChapters sim matrix size: 895, 895
VidChapters Text-to-ASR:
	>>>  R@1: 0.0 - R@5: 0.1 - R@10: 0.2 - Median R: 408.0 - Mean R: 407.8
VidChapters ASR-to-Text:
	>>>  Asr2T$R@1: 4.3 - Asr2T$R@5: 5.3 - Asr2T$R@10: 5.8 - Asr2T$Median R: 641.0 - Asr2T$Mean R: 541.1
VidChapters sim matrix size: 895, 895
VidChapters ASR-to-Video:
	>>>  R@1: 10.2 - R@5: 13.0 - R@10: 14.7 - Median R: 297.0 - Mean R: 341.6
VidChapters Video-to-ASR:
	>>>  V2Asr$R@1: 0.0 - V2Asr$R@5: 0.0 - V2Asr$R@10: 0.0 - V2Asr$Median R: 560.0 - V2Asr$Mean R: 559.5
"""

"""
VidChapters sim matrix size: 895, 895
	 Length-T: 895, Length-V:895
VidChapters Text-to-Video:
	>>>  R@1: 26.5 - R@5: 45.0 - R@10: 52.3 - Median R: 9.0 - Mean R: 110.0
VidChapters Video-to-Text:
	>>>  V2T$R@1: 9.6 - V2T$R@5: 17.6 - V2T$R@10: 21.2 - V2T$Median R: 58.0 - V2T$Mean R: 93.4
VidChapters sim matrix size: 895, 895
VidChapters Text-to-Audio:
	>>>  R@1: 0.6 - R@5: 3.5 - R@10: 6.3 - Median R: 236.0 - Mean R: 291.7
VidChapters Audio-to-Text:
	>>>  A2T$R@1: 0.6 - A2T$R@5: 1.8 - A2T$R@10: 2.9 - A2T$Median R: 174.0 - A2T$Mean R: 228.8
VidChapters sim matrix size: 895, 895
VidChapters Audio-to-Video:
	>>>  R@1: 4.9 - R@5: 18.0 - R@10: 29.9 - Median R: 31.0 - Mean R: 90.2
VidChapters Video-to-Audio:
	>>>  V2A$R@1: 4.1 - V2A$R@5: 19.2 - V2A$R@10: 30.1 - V2A$Median R: 33.0 - V2A$Mean R: 85.8
VidChapters sim matrix size: 895, 895
VidChapters Text-to-ASR:
	>>>  R@1: 29.0 - R@5: 39.2 - R@10: 44.6 - Median R: 25.0 - Mean R: 178.7
VidChapters ASR-to-Text:
	>>>  Asr2T$R@1: 16.0 - Asr2T$R@5: 21.7 - Asr2T$R@10: 23.6 - Asr2T$Median R: 359.0 - Asr2T$Mean R: 369.4
VidChapters sim matrix size: 895, 895
VidChapters ASR-to-Video:
	>>>  R@1: 26.4 - R@5: 47.0 - R@10: 57.9 - Median R: 6.0 - Mean R: 68.4
VidChapters Video-to-ASR:
	>>>  V2Asr$R@1: 21.4 - V2Asr$R@5: 39.1 - V2Asr$R@10: 47.6 - V2Asr$Median R: 13.0 - V2Asr$Mean R: 71.1
VidChapters sim matrix size: 895, 895
VidChapters Text-to-Summary_ASR:
	>>>  R@1: 23.7 - R@5: 35.0 - R@10: 39.9 - Median R: 46.0 - Mean R: 202.3
VidChapters Summary_ASR-to-Text:
	>>>  Sum2T$R@1: 13.1 - Sum2T$R@5: 19.4 - Sum2T$R@10: 21.9 - Sum2T$Median R: 443.0 - Sum2T$Mean R: 424.7
VidChapters sim matrix size: 895, 895
VidChapters Summary_ASR-to-Video:
	>>>  R@1: 26.3 - R@5: 46.4 - R@10: 57.9 - Median R: 7.0 - Mean R: 69.7
VidChapters Video-to-Summary_ASR:
	>>>  V2Sum$R@1: 19.1 - V2Sum$R@5: 36.6 - V2Sum$R@10: 44.6 - V2Sum$Median R: 16.0 - V2Sum$Mean R: 73.5
"""