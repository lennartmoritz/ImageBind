import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from easydict import EasyDict
from torch.utils.data import DataLoader
import numpy as np
import torch
import os.path as osp
from tqdm.auto import tqdm as tqdm
from datetime import datetime
import time
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from master_metrics import compute_metrics, calc_rand_retrieval_chance
import json
from vidchapters7m_dataloader import VidChapters7M_Dataset


def get_args_vidchap():
    # build args
    args = {
        "json_path": '/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/chapters_dvc_test.json',
        "video_folder": '/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/chapter_clips',
        "audio_folder": '/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/chapter_audios',
        "asr_folder": '/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/chapter_asr',
        "summary_folder": '/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/chapter_summary_asr',
        "report_folder": './reports/native/vidchap',
        "batch_size_val": 8,
        "num_thread_reader": 1,
        "cache_dir": '/ltstorage/home/1moritz/storage/models/languagebind/downloaded_weights',
    }
    args = EasyDict(args)
    return args


def run_eval(
        model: imagebind_model.ImageBindModel,
        dataloader: DataLoader,
        device: torch.device,
        report_folder: str = "./reports"
):
    batch_sentences_embeddings, batch_videos_embeddings, batch_audios_embeddings = [], [], []
    batch_summaries_embeddings, batch_asr_embeddings = [], []
    words_per_asr_chunk, asr_not_summarized = [], []
    words_per_summary, chunk_inference_times = [], []
    # Calculate embeddings
    for bid, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        sentences, video_paths, audio_paths, asr_texts, summaries, summary_stats = batch

        if not isinstance(sentences, list):
            sentences = list(sentences)
        if not isinstance(video_paths, list):
            video_paths = list(video_paths)
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
            start_time = time.perf_counter()
            embeddings = model(inputs)
            asr_embeddings = model(asr_inputs)
            inference_time = time.perf_counter() - start_time
            summary_embeddings = model(summary_inputs)

        batch_sentences_embeddings.append(embeddings[ModalityType.TEXT])
        batch_videos_embeddings.append(embeddings[ModalityType.VISION])
        batch_audios_embeddings.append(embeddings[ModalityType.AUDIO])
        batch_asr_embeddings.append(asr_embeddings[ModalityType.TEXT])
        batch_summaries_embeddings.append(summary_embeddings[ModalityType.TEXT])
        chunk_inference_times.append(inference_time)

        # update summary stats
        for n_words in summary_stats["n_words_asr"].tolist():
            # convert Tensor datatype native list to prevent crash when writing json report
            words_per_asr_chunk.append(n_words)
        for not_summarized in summary_stats["only_copy"].tolist():
            asr_not_summarized.append(not_summarized)
        for summary in summaries:
            words_per_summary.append(len(summary.split()))

    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_videos_embeddings)
    # Create match matrix
    sc_true_matches = np.arange(sim_matrix.shape[0]).reshape(-1, 1)
    cs_true_matches = np.arange(sim_matrix.T.shape[0]).reshape(-1, 1)
    # Log metrics Text-to-Video
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    tv_metrics = compute_metrics(sim_matrix, sc_true_matches)
    vt_metrics = compute_metrics(sim_matrix.T, cs_true_matches)
    print('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    print("VidChapters Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
          format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR'], tv_metrics['mAP']))
    print("VidChapters Video-to-Text:")
    print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f} - V2T$mAP: {:.2f}'.
          format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR'], vt_metrics['mAP']))

    # Log metrics Text-to-Audio
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_audios_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ta_metrics = compute_metrics(sim_matrix, sc_true_matches)
    at_metrics = compute_metrics(sim_matrix.T, cs_true_matches)
    print("VidChapters Text-to-Audio:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
          format(ta_metrics['R1'], ta_metrics['R5'], ta_metrics['R10'], ta_metrics['MR'], ta_metrics['MeanR'], ta_metrics['mAP']))
    print("VidChapters Audio-to-Text:")
    print('\t>>>  A2T$R@1: {:.1f} - A2T$R@5: {:.1f} - A2T$R@10: {:.1f} - A2T$Median R: {:.1f} - A2T$Mean R: {:.1f} - A2T$mAP: {:.2f}'.
          format(at_metrics['R1'], at_metrics['R5'], at_metrics['R10'], at_metrics['MR'], at_metrics['MeanR'], at_metrics['mAP']))

    # Log metrics Audio-to-Video
    sim_matrix = create_sim_matrix(batch_audios_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    av_metrics = compute_metrics(sim_matrix, np.arange(sim_matrix.shape[0]).reshape(-1, 1))
    va_metrics = compute_metrics(sim_matrix.T, np.arange(sim_matrix.T.shape[0]).reshape(-1, 1))
    print("VidChapters Audio-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
          format(av_metrics['R1'], av_metrics['R5'], av_metrics['R10'], av_metrics['MR'], av_metrics['MeanR'], av_metrics['mAP']))
    print("VidChapters Video-to-Audio:")
    print('\t>>>  V2A$R@1: {:.1f} - V2A$R@5: {:.1f} - V2A$R@10: {:.1f} - V2A$Median R: {:.1f} - V2A$Mean R: {:.1f} - V2A$mAP: {:.2f}'.
          format(va_metrics['R1'], va_metrics['R5'], va_metrics['R10'], va_metrics['MR'], va_metrics['MeanR'], va_metrics['mAP']))

    # Log metrics Text-to-ASR
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_asr_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    tasr_metrics = compute_metrics(sim_matrix, sc_true_matches)
    asrt_metrics = compute_metrics(sim_matrix.T, cs_true_matches)
    print("VidChapters Text-to-ASR:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
          format(tasr_metrics['R1'], tasr_metrics['R5'], tasr_metrics['R10'], tasr_metrics['MR'], tasr_metrics['MeanR'], tasr_metrics['mAP']))
    print("VidChapters ASR-to-Text:")
    print('\t>>>  Asr2T$R@1: {:.1f} - Asr2T$R@5: {:.1f} - Asr2T$R@10: {:.1f} - Asr2T$Median R: {:.1f} - Asr2T$Mean R: {:.1f} - Asr2T$mAP: {:.2f}'.
          format(asrt_metrics['R1'], asrt_metrics['R5'], asrt_metrics['R10'], asrt_metrics['MR'], asrt_metrics['MeanR'], asrt_metrics['mAP']))

    # Log metrics ASR-to-Video
    sim_matrix = create_sim_matrix(batch_asr_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    asrv_metrics = compute_metrics(sim_matrix, np.arange(sim_matrix.shape[0]).reshape(-1, 1))
    vasr_metrics = compute_metrics(sim_matrix.T, np.arange(sim_matrix.T.shape[0]).reshape(-1, 1))
    print("VidChapters ASR-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
          format(asrv_metrics['R1'], asrv_metrics['R5'], asrv_metrics['R10'], asrv_metrics['MR'], asrv_metrics['MeanR'], asrv_metrics['mAP']))
    print("VidChapters Video-to-ASR:")
    print('\t>>>  V2Asr$R@1: {:.1f} - V2Asr$R@5: {:.1f} - V2Asr$R@10: {:.1f} - V2Asr$Median R: {:.1f} - V2Asr$Mean R: {:.1f} - V2Asr$mAP: {:.2f}'.
          format(vasr_metrics['R1'], vasr_metrics['R5'], vasr_metrics['R10'], vasr_metrics['MR'], vasr_metrics['MeanR'], vasr_metrics['mAP']))

    # Log metrics Text-to-Summary
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_summaries_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ts_metrics = compute_metrics(sim_matrix, sc_true_matches)
    st_metrics = compute_metrics(sim_matrix.T, cs_true_matches)
    print("VidChapters Text-to-Summary_ASR:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
          format(ts_metrics['R1'], ts_metrics['R5'], ts_metrics['R10'], ts_metrics['MR'], ts_metrics['MeanR'], ts_metrics['mAP']))
    print("VidChapters Summary_ASR-to-Text:")
    print('\t>>>  Sum2T$R@1: {:.1f} - Sum2T$R@5: {:.1f} - Sum2T$R@10: {:.1f} - Sum2T$Median R: {:.1f} - Sum2T$Mean R: {:.1f} - Sum2T$mAP: {:.2f}'.
          format(st_metrics['R1'], st_metrics['R5'], st_metrics['R10'], st_metrics['MR'], st_metrics['MeanR'], st_metrics['mAP']))

    # Log metrics Summary-to-Video
    sim_matrix = create_sim_matrix(batch_summaries_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    sv_metrics = compute_metrics(sim_matrix, np.arange(sim_matrix.shape[0]).reshape(-1, 1))
    vs_metrics = compute_metrics(sim_matrix.T, np.arange(sim_matrix.T.shape[0]).reshape(-1, 1))
    print("VidChapters Summary_ASR-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
          format(sv_metrics['R1'], sv_metrics['R5'], sv_metrics['R10'], sv_metrics['MR'], sv_metrics['MeanR'], sv_metrics['mAP']))
    print("VidChapters Video-to-Summary_ASR:")
    print('\t>>>  V2Sum$R@1: {:.1f} - V2Sum$R@5: {:.1f} - V2Sum$R@10: {:.1f} - V2Sum$Median R: {:.1f} - V2Sum$Mean R: {:.1f} - V2Sum$mAP: {:.2f}'.
          format(vs_metrics['R1'], vs_metrics['R5'], vs_metrics['R10'], vs_metrics['MR'], vs_metrics['MeanR'], vs_metrics['mAP']))

    chances = calc_rand_retrieval_chance(sc_true_matches, cs_true_matches)
    report = {
        "dimensions": {"sentences": len(sc_true_matches), "chunks": len(cs_true_matches)},
        "avg_c_inference_time": sum(chunk_inference_times) / len(cs_true_matches),
        "summaries": create_summary_report(words_per_asr_chunk, asr_not_summarized, words_per_summary),
        "chances": {
            "s2c_chance": chances.s2c_chance,
            "c2s_chance": chances.c2s_chance,
            "c2c_chance": chances.c2c_chance,
            "s2s_chance": chances.s2s_chance,
        },
        "text2video": tv_metrics,
        "video2text": vt_metrics,
        "text2audio": ta_metrics,
        "audio2text": at_metrics,
        "audio2video": av_metrics,
        "video2audio": va_metrics,
        "text2asr": tasr_metrics,
        "asr2text": asrt_metrics,
        "asr2video": asrv_metrics,
        "video2asr": vasr_metrics,
        "text2summary": ts_metrics,
        "summary2text": st_metrics,
        "summary2video": sv_metrics,
        "video2summary": vs_metrics,
    }
    json_report = json.dumps(report, indent=4)
    os.makedirs(report_folder, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    with open(osp.join(report_folder, f"report_eval_{current_time}.json"), "w") as outfile:
        outfile.write(json_report)


def create_summary_report(words_per_asr_chunk, asr_not_summarized, words_per_summary):
    n_chunks = len(words_per_asr_chunk)
    n_summaries = n_chunks - sum(asr_not_summarized)
    summaries = {
        "n_total": n_chunks,
        "n_summarized": n_summaries,
        "summary_quota": n_summaries / n_chunks,
        "raw_asr": {
            "avg_words": sum(words_per_asr_chunk) / n_chunks,
            "min_words": min(words_per_asr_chunk),
            "max_words": max(words_per_asr_chunk),
        },
        "summarized_asr": {
            "avg_words": sum(words_per_summary) / n_chunks,
            "min_words": min(words_per_summary),
            "max_words": max(words_per_summary),
        }
    }
    return summaries


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
            b1b2 = sequence_output @ visual_output.T
            b1b2 = b1b2.cpu().detach().numpy()
            each_row.append(b1b2)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    return sim_matrix


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

    return run_eval(model, dataloader_vidchap, device, args.report_folder)


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
	>>>  R@1: 26.5 - R@5: 45.0 - R@10: 52.3 - Median R: 9.0 - Mean R: 110.0 - mAP: 0.4
VidChapters Video-to-Text:
	>>>  V2T$R@1: 23.1 - V2T$R@5: 40.8 - V2T$R@10: 46.1 - V2T$Median R: 17.0 - V2T$Mean R: 103.8 - V2T$mAP: 0.3
VidChapters sim matrix size: 895, 895
VidChapters Text-to-Audio:
	>>>  R@1: 0.6 - R@5: 3.5 - R@10: 6.3 - Median R: 236.0 - Mean R: 291.7 - mAP: 0.0
VidChapters Audio-to-Text:
	>>>  A2T$R@1: 1.2 - A2T$R@5: 3.8 - A2T$R@10: 5.7 - A2T$Median R: 250.0 - A2T$Mean R: 300.4 - A2T$mAP: 0.0
VidChapters sim matrix size: 895, 895
VidChapters Audio-to-Video:
	>>>  R@1: 4.9 - R@5: 18.0 - R@10: 29.9 - Median R: 31.0 - Mean R: 90.2 - mAP: 0.1
VidChapters Video-to-Audio:
	>>>  V2A$R@1: 4.1 - V2A$R@5: 19.2 - V2A$R@10: 30.1 - V2A$Median R: 33.0 - V2A$Mean R: 85.8 - V2A$mAP: 0.1
VidChapters sim matrix size: 895, 895
VidChapters Text-to-ASR:
	>>>  R@1: 32.7 - R@5: 44.2 - R@10: 49.5 - Median R: 13.0 - Mean R: 157.8 - mAP: 0.4
VidChapters ASR-to-Text:
	>>>  Asr2T$R@1: 38.7 - Asr2T$R@5: 52.2 - Asr2T$R@10: 55.5 - Asr2T$Median R: 4.0 - Asr2T$Mean R: 160.9 - Asr2T$mAP: 0.4
VidChapters sim matrix size: 895, 895
VidChapters ASR-to-Video:
	>>>  R@1: 26.4 - R@5: 47.0 - R@10: 57.9 - Median R: 6.0 - Mean R: 68.4 - mAP: 0.4
VidChapters Video-to-ASR:
	>>>  V2Asr$R@1: 24.1 - V2Asr$R@5: 44.1 - V2Asr$R@10: 53.6 - V2Asr$Median R: 8.0 - V2Asr$Mean R: 65.5 - V2Asr$mAP: 0.3
VidChapters sim matrix size: 895, 895
VidChapters Text-to-Summary_ASR:
	>>>  R@1: 26.7 - R@5: 39.3 - R@10: 44.1 - Median R: 28.0 - Mean R: 184.7 - mAP: 0.3
VidChapters Summary_ASR-to-Text:
	>>>  Sum2T$R@1: 31.7 - Sum2T$R@5: 46.6 - Sum2T$R@10: 52.1 - Sum2T$Median R: 8.0 - Sum2T$Mean R: 180.7 - Sum2T$mAP: 0.4
VidChapters sim matrix size: 895, 895
VidChapters Summary_ASR-to-Video:
	>>>  R@1: 26.3 - R@5: 46.4 - R@10: 57.9 - Median R: 7.0 - Mean R: 69.7 - mAP: 0.4
VidChapters Video-to-Summary_ASR:
	>>>  V2Sum$R@1: 21.6 - V2Sum$R@5: 41.2 - V2Sum$R@10: 50.3 - V2Sum$Median R: 10.0 - V2Sum$Mean R: 70.2 - V2Sum$mAP: 0.3
"""