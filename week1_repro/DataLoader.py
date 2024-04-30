import json
from pathlib import Path
import faiss

import numpy as np
import torch
from torch.utils.data import Dataset as PyTorchDataset

from labml import lab, monit
from labml_helpers.datasets.text import TextFileDataset, TextDataset
from labml_nn.transformers.retro.database import RetroIndex
from labml_nn.transformers.retro.bert_embeddings import BERTChunkEmbeddings

def build_database(chunk_len: int = 16, batch_size: int = 64, d_emb: int = 768, n_centeroids: int = 256,
                   code_size: int = 64, n_probe: int = 8, n_train: int = 50_000):

    # Load the dataset text file
    dataset = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    # Get training data (a string)
    text = dataset.train

    # Split the text into chunks of `chunk_length`
    chunks = [text[i:i + chunk_len] for i in range(0, len(text), chunk_len) if i + chunk_len * 2 < len(text)]
    # Get the offsets of each of the chunks
    chunk_offsets = np.array([i for i in range(0, len(text), chunk_len) if i + chunk_len * 2 < len(text)])
    # Number of chunks
    n_chunks = len(chunks)

    # Initialize BERT to get $\text{B\small{ERT}}(N)$
    bert = BERTChunkEmbeddings(torch.device('cuda:2'))

    # Get chunk embeddings by processing `batch_size` number of chunks on each iteration
    chunk_emb = []
    for i in monit.iterate('Get embeddings', range(0, n_chunks, batch_size)):
        chunk_emb.append(bert(chunks[i: i + batch_size]).cpu())
    # Merge them into a single tensor
    chunk_emb = torch.cat(chunk_emb, dim=0).numpy()

    # Create the [FAISS index](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFPQ.html)
    quantizer = faiss.IndexFlatL2(d_emb)
    index = faiss.IndexIVFPQ(quantizer, d_emb, n_centeroids, code_size, 8)
    index.nprobe = n_probe

    # Get a random sample of the the chunk indexes
    random_sample = np.random.choice(np.arange(n_chunks), size=[min(n_train, n_chunks)], replace=False)

    # Train the index to store the keys
    with monit.section('Train index'):
        index.train(chunk_emb[random_sample])

    # Add the chunks to the index in batches of size `1024`
    for s in monit.iterate('Index', range(0, n_chunks, 1024)):
        e = min(s + 1024, n_chunks)
        # Add to index
        index.add_with_ids(chunk_emb[s:e], chunk_offsets[s: e])

    # Save the index
    with monit.section('Save'):
        faiss.write_index(index, str(lab.get_data_path() / 'retro.index'))


def build_dataset(chunk_len: int = 16, chunks_per_sample: int = 16, skip_range: int = 8):

    # Load the text file
    dataset = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    # Training portion of it
    text = dataset.train

    # Load the index for retrieving neighbors
    index = RetroIndex(n_neighbors=2)

    # The input sample offsets
    sample_offsets = []
    # Cursor for the text
    i = 0
    while i < len(text):
        # Skip a few characters to make sure it's not aligned with the neighbors
        skip = np.random.randint(skip_range)
        i += skip

        # Stop if we've reached the end of the text
        if i + chunks_per_sample * chunk_len > len(text):
            break

        # Collect the offset
        sample_offsets.append(i)

        # Increment the cursor
        i += chunks_per_sample * chunk_len

    # For samples
    samples = []
    # Iterate through sample offsets
    for i in monit.iterate('Gather Neighbors', sample_offsets):
        # Get the sample including an extra character (for prediction)
        sample = text[i: i + chunks_per_sample * chunk_len + 1]
        # The input
        src = sample[:-1]
        # Break it into chunks
        chunks = [src[j:j + chunk_len] for j in range(0, len(src), chunk_len)]
        # The chunk offsets
        chunk_offsets = [j + i for j in range(0, len(src), chunk_len)]

        # Retrieve nearest neighbors
        neighbor_offsets = index(chunks, chunk_offsets)

        # Get neighbor texts. The neighbor length is twice the `chunk_len`
        neighbors = [[text[j: j + chunk_len] for j in n_off] for n_off in neighbor_offsets]

        # Add to list of samples
        samples.append((sample[:-1], sample[1:], neighbors))

    # Save the samples in JSON.
    # We don't need to use complex dataset storage mechanisms or pre-tokenize
    # since our dataset is small.
    with open(str(lab.get_data_path() / 'retro_train_dataset.json'), 'w') as f:
        f.write(json.dumps(samples))


class Dataset(PyTorchDataset):
    """
    ## Dataset

    This is the PyTorch dataset that loads the dataset created
    by `build_dataset`.
    """
    def __init__(self, file_path: Path, tds: TextDataset):
        """
        * `file_path` is the path of the saved JSON file
        * `tds` is the `TextDataset`
        """

        self.tds = tds
        # Load the samples
        with open(str(file_path), 'r') as f:
            self.samples = json.loads(f.read())

    def __len__(self):
        """
        Number of samples
        """
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Get a sample
        """
        # Get the sample
        s = self.samples[idx]
        # Tokenize
        src = self.tds.text_to_i(s[0])
        tgt = self.tds.text_to_i(s[1])
        neighbors = torch.stack([torch.stack([self.tds.text_to_i(n) for n in chunks]) for chunks in s[2]])
        #
        return src, tgt, neighbors

#
if __name__ == '__main__':
    build_database()
    build_dataset()