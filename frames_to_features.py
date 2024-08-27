import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from tqdm.auto import tqdm
import os.path as osp
import numpy as np
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from sklearn.metrics.pairwise import cosine_similarity


def main(frame_folder, out_feature_folder, batch_size):
    assert torch.cuda.is_available()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    torch_features = frames_to_features(model, frame_folder, batch_size, device)
    numpy_features = torch_features.cpu().numpy()

    # Define the output path
    feature_out_path = osp.join(out_feature_folder, "features.npz")
    if not os.path.exists(out_feature_folder):
        os.makedirs(out_feature_folder)

    # Save the numpy array as a compressed .npz file
    np.savez_compressed(feature_out_path, features=numpy_features)
    print(f"Features saved to {feature_out_path}")
    print("Feature matrix shape:", numpy_features.shape)

    loaded_features = load_features(feature_out_path)
    similarity_matrix = cosine_similarity(loaded_features)
    print("Similarity matrix shape:", similarity_matrix.shape)


def frames_to_features(model, frame_folder, batch_size, device):
    # List all .jpg files in the directory
    frames = [f for f in os.listdir(frame_folder) if f.lower().endswith('.jpg')]
    frames.sort(key=lambda x: int(osp.splitext(x)[0]))  # Sort by frame number

    features = None  # Initialize as None for the first concatenation
    for i in tqdm(range(0, len(frames), batch_size)):
        frames_batch = frames[i: i + batch_size]
        frames_batch = [osp.join(frame_folder, f_name) for f_name in frames_batch]

        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(frames_batch, device)
        }

        with torch.no_grad():
            embeddings = model(inputs)

        if features is None:
            features = embeddings[ModalityType.VISION]  # Initialize features with the first batch
        else:
            # Concatenate along the batch dimension
            features = torch.cat((features, embeddings[ModalityType.VISION]), dim=0)
    return features


def load_features(feature_file):
    # Load the numpy array from the .npz file
    with np.load(feature_file) as data:
        features = data['features']
    return features


if __name__ == '__main__':
    # vid_id = "098sLWluu88"
    vid_id = "x9V3ccne4k8"
    main(
        frame_folder=f"/ltstorage/home/1moritz/Repositories/WhisperASR/Frames/{vid_id}",
        out_feature_folder=f"/ltstorage/home/1moritz/Repositories/WhisperASR/Features/{vid_id}",
        batch_size=32
    )
