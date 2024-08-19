import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import numpy as np
from easydict import EasyDict
import os.path as osp
import torch
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as tqdm
from datetime import datetime
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from chunk_vidchap_dataloader import VidChapters7M_Dataset, VidChapText_Dataset
from master_metrics import compute_metrics


def get_args_vidchap(fixed_length, chunking_mode):
    assert isinstance(fixed_length, int)
    assert chunking_mode in ["fixed", "recursive", "semantic"]
    # build args
    args = {
        "json_path": '/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/chapters_dvc_test.json',
        "video_folder": f'/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/chunking/{chunking_mode}/{fixed_length}s/clips',
        "audio_folder": f'/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/chunking/{chunking_mode}/{fixed_length}s/audios',
        "asr_folder": f'/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/chunking/{chunking_mode}/{fixed_length}s/asr',
        "summary_folder": f'/ltstorage/home/1moritz/storage/datasets/VidChapters-7M/chunking/{chunking_mode}/{fixed_length}s/summary_asr',
        "report_folder": f'./reports/{chunking_mode}/{fixed_length}s',
        "batch_size_val": 8,
        "num_thread_reader": 1,
        "cache_dir": '/ltstorage/home/1moritz/storage/models/languagebind/downloaded_weights',
    }
    args = EasyDict(args)
    return args

def run_eval(
        model: imagebind_model.ImageBindModel,
        chunk_dataloader: DataLoader,
        sentence_dataloader: DataLoader,
        device: torch.device,
        report_folder: str = "./reports"
):
    batch_videos_embeddings, batch_audios_embeddings, batch_asr_embeddings = [], [], []
    batch_summaries_embeddings, own_chunk_ids, target_sentence_ids = [], [], []
    # Calculate embeddings
    for bid, batch in tqdm(enumerate(chunk_dataloader), total=len(chunk_dataloader)):
        chunk_ids, sentence_ids, video_paths, audio_paths, asr_texts, summaries = batch

        if not isinstance(chunk_ids, list):
            chunk_ids = [chunk_ids]
        if not isinstance(sentence_ids, list):
            sentence_ids = [sentence_ids]
        if not isinstance(video_paths, list):
            video_paths = list(video_paths)
        if not isinstance(audio_paths, list):
            audio_paths = list(audio_paths)
        if not isinstance(asr_texts, list):
            asr_texts = list(asr_texts)
        if not isinstance(summaries, list):
            summaries = list(summaries)

        # Re-create sentence_ids and chunk_ids as tuples, since tuples were broken apart into 
        # two batch sized lists containing only the first or second parts of the identifiers
        assert len(sentence_ids) == 2
        sentence_ids = [identifier for identifier in zip(sentence_ids[0], sentence_ids[1].tolist())]
        assert len(chunk_ids) == 2
        chunk_ids = [identifier for identifier in zip(chunk_ids[0], chunk_ids[1].tolist())]

        # Load data
        inputs = {
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

        batch_videos_embeddings.append(embeddings[ModalityType.VISION])
        batch_audios_embeddings.append(embeddings[ModalityType.AUDIO])
        batch_asr_embeddings.append(asr_embeddings[ModalityType.TEXT])
        batch_summaries_embeddings.append(summary_embeddings[ModalityType.TEXT])
        own_chunk_ids.append(chunk_ids)
        target_sentence_ids.append(sentence_ids)

    own_chunk_ids = np.concatenate(tuple(own_chunk_ids), axis=0).tolist()
    own_chunk_ids = [tuple((item[0], int(item[1]))) for item in own_chunk_ids]
    target_sentence_ids = np.concatenate(tuple(target_sentence_ids), axis=0).tolist()
    target_sentence_ids = [tuple((item[0], int(item[1]))) for item in target_sentence_ids]

    batch_sentences_embeddings, own_sentence_ids, target_chunk_ids = [], [], []
    for bid, batch in tqdm(enumerate(sentence_dataloader), total=len(sentence_dataloader)):
        sentence_ids, chunk_ids, sentences = batch

        if not isinstance(sentence_ids, list):
            sentence_ids = [sentence_ids]
        if not isinstance(chunk_ids, list):
            chunk_ids = [chunk_ids]
        if not isinstance(sentences, list):
            sentences = list(sentences)

        # Re-create sentence_ids and chunk_ids as tuples, since tuples were broken apart into 
        # two batch sized lists containing only the first or second parts of the identifiers
        assert len(sentence_ids) == 2
        sentence_ids = [identifier for identifier in zip(sentence_ids[0], sentence_ids[1].tolist())]
        assert len(chunk_ids) == 2
        chunk_ids = [identifier for identifier in zip(chunk_ids[0], chunk_ids[1].tolist())]

        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(sentences, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        batch_sentences_embeddings.append(embeddings[ModalityType.TEXT])
        own_sentence_ids.append(sentence_ids)
        target_chunk_ids.append(chunk_ids)

    own_sentence_ids = np.concatenate(tuple(own_sentence_ids), axis=0).tolist()
    own_sentence_ids = [tuple((item[0], int(item[1]))) for item in own_sentence_ids]
    target_chunk_ids = np.concatenate(tuple(target_chunk_ids), axis=0).tolist()
    target_chunk_ids = [tuple((item[0], int(item[1]))) for item in target_chunk_ids]

    # Create match matrix
    sc_true_matches = create_match_matrix(own_sentence_ids, target_chunk_ids, own_chunk_ids, target_sentence_ids)
    cs_true_matches = create_match_matrix(own_chunk_ids, target_sentence_ids, own_sentence_ids, target_chunk_ids)

    # Log metrics Text-to-Video
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    tv_metrics = compute_metrics(sim_matrix, sc_true_matches)
    vt_metrics = compute_metrics(sim_matrix.T, cs_true_matches)
    print(f"VidChapters Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR'], tv_metrics['mAP']))
    print(f"VidChapters Video-to-Text:")
    print('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f} - V2T$mAP: {:.2f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR'], vt_metrics['mAP']))

    # Log metrics Text-to-Audio
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_audios_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ta_metrics = compute_metrics(sim_matrix, sc_true_matches)
    at_metrics = compute_metrics(sim_matrix.T, cs_true_matches)
    print(f"VidChapters Text-to-Audio:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
                format(ta_metrics['R1'], ta_metrics['R5'], ta_metrics['R10'], ta_metrics['MR'], ta_metrics['MeanR'], ta_metrics['mAP']))
    print(f"VidChapters Audio-to-Text:")
    print('\t>>>  A2T$R@1: {:.1f} - A2T$R@5: {:.1f} - A2T$R@10: {:.1f} - A2T$Median R: {:.1f} - A2T$Mean R: {:.1f} - A2T$mAP: {:.2f}'.
                format(at_metrics['R1'], at_metrics['R5'], at_metrics['R10'], at_metrics['MR'], at_metrics['MeanR'], at_metrics['mAP']))

    # Log metrics Audio-to-Video
    sim_matrix = create_sim_matrix(batch_audios_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    av_metrics = compute_metrics(sim_matrix, np.arange(sim_matrix.shape[0]).reshape(-1, 1))
    va_metrics = compute_metrics(sim_matrix.T, np.arange(sim_matrix.T.shape[0]).reshape(-1, 1))
    print(f"VidChapters Audio-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
                format(av_metrics['R1'], av_metrics['R5'], av_metrics['R10'], av_metrics['MR'], av_metrics['MeanR'], av_metrics['mAP']))
    print(f"VidChapters Video-to-Audio:")
    print('\t>>>  V2A$R@1: {:.1f} - V2A$R@5: {:.1f} - V2A$R@10: {:.1f} - V2A$Median R: {:.1f} - V2A$Mean R: {:.1f} - V2A$mAP: {:.2f}'.
                format(va_metrics['R1'], va_metrics['R5'], va_metrics['R10'], va_metrics['MR'], va_metrics['MeanR'], va_metrics['mAP']))

    # Log metrics Text-to-ASR
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_asr_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    tasr_metrics = compute_metrics(sim_matrix, sc_true_matches)
    asrt_metrics = compute_metrics(sim_matrix.T, cs_true_matches)
    print(f"VidChapters Text-to-ASR:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
                format(tasr_metrics['R1'], tasr_metrics['R5'], tasr_metrics['R10'], tasr_metrics['MR'], tasr_metrics['MeanR'], tasr_metrics['mAP']))
    print(f"VidChapters ASR-to-Text:")
    print('\t>>>  Asr2T$R@1: {:.1f} - Asr2T$R@5: {:.1f} - Asr2T$R@10: {:.1f} - Asr2T$Median R: {:.1f} - Asr2T$Mean R: {:.1f} - Asr2T$mAP: {:.2f}'.
                format(asrt_metrics['R1'], asrt_metrics['R5'], asrt_metrics['R10'], asrt_metrics['MR'], asrt_metrics['MeanR'], asrt_metrics['mAP']))

    # Log metrics ASR-to-Video
    sim_matrix = create_sim_matrix(batch_asr_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    asrv_metrics = compute_metrics(sim_matrix, np.arange(sim_matrix.shape[0]).reshape(-1, 1))
    vasr_metrics = compute_metrics(sim_matrix.T, np.arange(sim_matrix.T.shape[0]).reshape(-1, 1))
    print(f"VidChapters ASR-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
                format(asrv_metrics['R1'], asrv_metrics['R5'], asrv_metrics['R10'], asrv_metrics['MR'], asrv_metrics['MeanR'], asrv_metrics['mAP']))
    print(f"VidChapters Video-to-ASR:")
    print('\t>>>  V2Asr$R@1: {:.1f} - V2Asr$R@5: {:.1f} - V2Asr$R@10: {:.1f} - V2Asr$Median R: {:.1f} - V2Asr$Mean R: {:.1f} - V2Asr$mAP: {:.2f}'.
                format(vasr_metrics['R1'], vasr_metrics['R5'], vasr_metrics['R10'], vasr_metrics['MR'], vasr_metrics['MeanR'], vasr_metrics['mAP']))

    # Log metrics Text-to-Summary
    sim_matrix = create_sim_matrix(batch_sentences_embeddings, batch_summaries_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    ts_metrics = compute_metrics(sim_matrix, sc_true_matches)
    st_metrics = compute_metrics(sim_matrix.T, cs_true_matches)
    print(f"VidChapters Text-to-Summary_ASR:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
                format(ts_metrics['R1'], ts_metrics['R5'], ts_metrics['R10'], ts_metrics['MR'], ts_metrics['MeanR'], ts_metrics['mAP']))
    print(f"VidChapters Summary_ASR-to-Text:")
    print('\t>>>  Sum2T$R@1: {:.1f} - Sum2T$R@5: {:.1f} - Sum2T$R@10: {:.1f} - Sum2T$Median R: {:.1f} - Sum2T$Mean R: {:.1f} - Sum2T$mAP: {:.2f}'.
                format(st_metrics['R1'], st_metrics['R5'], st_metrics['R10'], st_metrics['MR'], st_metrics['MeanR'], st_metrics['mAP']))

    # Log metrics Summary-to-Video
    sim_matrix = create_sim_matrix(batch_summaries_embeddings, batch_videos_embeddings)
    print(f"VidChapters sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
    sv_metrics = compute_metrics(sim_matrix, np.arange(sim_matrix.shape[0]).reshape(-1, 1))
    vs_metrics = compute_metrics(sim_matrix.T, np.arange(sim_matrix.T.shape[0]).reshape(-1, 1))
    print(f"VidChapters Summary_ASR-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f} - mAP: {:.2f}'.
                format(sv_metrics['R1'], sv_metrics['R5'], sv_metrics['R10'], sv_metrics['MR'], sv_metrics['MeanR'], sv_metrics['mAP']))
    print(f"VidChapters Video-to-Summary_ASR:")
    print('\t>>>  V2Sum$R@1: {:.1f} - V2Sum$R@5: {:.1f} - V2Sum$R@10: {:.1f} - V2Sum$Median R: {:.1f} - V2Sum$Mean R: {:.1f} - V2Sum$mAP: {:.2f}'.
                format(vs_metrics['R1'], vs_metrics['R5'], vs_metrics['R10'], vs_metrics['MR'], vs_metrics['MeanR'], vs_metrics['mAP']))

    report = {
        "dimensions": {"sentences": len(sc_true_matches), "chunks": len(cs_true_matches)},
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

def create_match_matrix(own_query_ids, target_samples_ids, own_samples_ids, target_query_ids):
    """
    Return list of lists containing matching indexes from the samples for a given query.

    Args:
        own_query_ids:          Maintains the order of query identifiers.
        target_samples_ids:     Denotes the max IoU targeted sample for each query in order.
        own_samples_ids:        Maintains the order of sample identifiers.
        target_query_ids:       Denotes the max IoU targeted query for each sample in order.
    """
    true_matches = []
    for i, own_sentence_id in enumerate(own_query_ids):
        current_matches = []
        current_matches.append(own_samples_ids.index(target_samples_ids[i]))
        for j, sentence_id in enumerate(target_query_ids):
            if sentence_id == own_sentence_id:
                current_matches.append(j)
        current_matches = list(set(current_matches))
        assert len(current_matches) >= 1
        true_matches.append(current_matches)
    return true_matches

def main(args):
    assert torch.cuda.is_available()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

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

    dataloader_sentences = DataLoader(
        VidChapText_Dataset(json_path=args.json_path, video_folder=args.video_folder),
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )

    return run_eval(model, dataloader_vidchap, dataloader_sentences, device, args.report_folder)


if __name__ == '__main__':
    lengths = [15, 25, 50, 100, 150]
    c_mode = "fixed"
    for length in lengths[3:]:
        print(f"Start eval: Length = {length}s, Chunking = {c_mode}")
        arguments = get_args_vidchap(fixed_length=length, chunking_mode=c_mode)
        main(args=arguments)

    c_mode = "recursive"
    for length in lengths:
        print(f"Start eval: Length = {length}s, Chunking = {c_mode}")
        arguments = get_args_vidchap(fixed_length=length, chunking_mode=c_mode)
        main(args=arguments)
