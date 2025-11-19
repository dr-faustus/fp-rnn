import importlib
import os
import pickle
import random
import torch

from torch.utils.data import RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

catbAbI10k_TEMPLATE = "en-valid-10k_{}.txt"
catbAbI1k_TEMPLATE = "en-valid_{}.txt"
PARTITIONS = ["train", "valid", "test"]
DATA_PATH = "data/catbAbI_v1.2"
PAD_STR = "<pad>"
END_OF_STORY = "<eos>"
LARGE_10K = "catbAbI10k"
SMALL_1K = "catbAbI1k"


def load_default_params(p):
    p["dataset_variation"] = LARGE_10K
    p["num_workers"] = 2
    # Train on selected tasks by "t" e.g. "3t6t5t12" to train  on 2, 6, 5, and 12
    # empty string trains on all tasks
    p["whitelist"] = ""  
    # request answer mode -> only predict answers otherwise everything
    p["ra_mode"] = True
    p["seq_len"] = 200


def get_string_description(p):
  txt = "{}_{}{}"
  if len(p["whitelist"]) == 0:
    return txt.format(p["dataset_variation"],
                      "raMode" if p["ra_mode"] else "lmMode",
                      f"_sl{p['seq_len']}",
                      "")
  else:
    whitelist_str = "_task" + "-".join(sorted(set(p["whitelist"].split("t"))))
    return txt.format(p["dataset_variation"],
                      "raMode" if p["ra_mode"] else "lmMode",
                      f"_sl{p['seq_len']}",
                      whitelist_str)


def create_iterator(args, partition, batch_size, random=True):
  is_large = args.dataset_variation == LARGE_10K
  dataset = catbAbI(partition=partition,
                    whitelist=args.whitelist,
                    ra_mode=args.ra_mode,
                    large=is_large)
  args.PAD = dataset.word2idx[PAD_STR]
  args.EOS = dataset.word2idx[END_OF_STORY]
  args.QM = dataset.word2idx["?"]

  # create data loader
  if random:
    sampler = RandomSampler(dataset, replacement=False)
  else:
    sampler = SequentialSampler(dataset)
  batch_generator = StoryBatcher(sampler,
                                 batch_size=batch_size,
                                 seq_len=args.seq_len,
                                 PAD=args.PAD)
  return batch_generator


def read_samples(file_path, word2idx, whitelist):
  samples = []
  with open(file_path, "r") as f:
    for line in f:
      task, story = line.rstrip('\n').split("\t")
      if str(task) in whitelist.split("t") or len(whitelist) == 0:
        words = story.split(" ")
        # encode samples
        EOS = word2idx[END_OF_STORY]
        x = [EOS] + [word2idx[word] for word in words]
        y = [word2idx[word] for word in words] + [EOS]
        t = [int(task)] * len(x)
        samples.append((x, y, t))
  return samples


class catbAbI(data.Dataset):
  def __init__(self, partition, whitelist,
               ra_mode, large=True, folder=DATA_PATH):
    self.partition = partition
    self.whitelist = whitelist
    self.ra_mode = ra_mode

    if large:
      self.fp = os.path.join(folder, catbAbI10k_TEMPLATE.format(partition))
    else:
      self.fp = os.path.join(folder, catbAbI1k_TEMPLATE.format(partition))

    with open(os.path.join(folder, "vocab.pkl"), "rb") as f:
      self.word2idx, self.idx2word = pickle.load(f)

    self.samples = read_samples(self.fp, self.word2idx, self.whitelist)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    x, y, t = self.samples[index]
    x = torch.tensor(x).long()
    y = torch.tensor(y).long()
    t = torch.tensor(t).long()

    if self.ra_mode:
      qm_pos = x != self.word2idx["?"]
      y[qm_pos] = self.word2idx[PAD_STR]

    return x, y, t, len(x)


class StoryBatcher:
  def __init__(self, sampler, batch_size, seq_len, PAD, buffer_size=None):
    super().__init__()
    self.sampler = sampler
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.PAD = PAD
    self.dataset = sampler.data_source
    if buffer_size:
      self.buffer_size = buffer_size
    else:
      self.buffer_size = seq_len * 4
    self.buffer_x = [torch.tensor([]).long()] * batch_size
    self.buffer_y = [torch.tensor([]).long()] * batch_size
    self.buffer_t = [torch.tensor([]).long()] * batch_size

  def __iter__(self):
    self.sampler_iter = iter(self.sampler)
    return self

  def __next__(self):
    # fill buffer
    while True:
      lengths = [len(t) for t in self.buffer_x]
      min_len_idx = lengths.index(min(lengths))

      if min(lengths) >= self.buffer_size:
        break

      idx = next(self.sampler_iter, None)
      if idx is None:
        break
      else:
        x, y, t, length = self.sampler.data_source[idx]
        self.buffer_x[min_len_idx] = torch.cat([self.buffer_x[min_len_idx], x])
        self.buffer_y[min_len_idx] = torch.cat([self.buffer_y[min_len_idx], y])
        self.buffer_t[min_len_idx] = torch.cat([self.buffer_t[min_len_idx], t])

      # lengths = [len(t) for t in self.buffer_x]
      # print("lengths: ", lengths)

    if sum(lengths) == 0:
      raise StopIteration

    # get a batch
    batch_x = [b[:self.seq_len] for b in self.buffer_x]
    batch_y = [b[:self.seq_len] for b in self.buffer_y]
    batch_t = [b[:self.seq_len] for b in self.buffer_t]
    batch_len = torch.tensor([len(x) for x in batch_x])

    # pop from buffer
    self.buffer_x = [b[self.seq_len:] for b in self.buffer_x]
    self.buffer_y = [b[self.seq_len:] for b in self.buffer_y]
    self.buffer_t = [b[self.seq_len:] for b in self.buffer_t]

    # pad into tensor
    x_pad = pad_sequence(batch_x, batch_first=True, padding_value=self.PAD)
    y_pad = pad_sequence(batch_y, batch_first=True, padding_value=self.PAD)
    t_pad = pad_sequence(batch_t, batch_first=True, padding_value=self.PAD)

    return x_pad, y_pad, t_pad, batch_len

  def __len__(self):
    # approximate
    words = sum([len(sample[0]) for sample in self.sampler.data_source])
    return round(words / (self.batch_size * self.seq_len) + 0.5)




