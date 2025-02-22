import os
import random
import numpy as np
import torch


def set_seed(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('Set random seed as {} for pytorch'.format(seed))


def standardize_label(factor_input):
    factor_output = (factor_input - np.nanmean(factor_input)) / np.nanstd(factor_input)
    return factor_output


def pearson_r_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym)
    x_square_sum = torch.sum(xm * xm)
    y_square_sum = torch.sum(ym * ym)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return -torch.mean(r)


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym)
    x_square_sum = torch.sum(xm * xm)
    y_square_sum = torch.sum(ym * ym)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return torch.mean(r)


def load_train_data(x1, y, seq_len=40, step=5):

    x1_in_sample = np.zeros(
        (int(x1.shape[0] * x1.shape[1] / step), seq_len, x1.shape[2]))
    y_in_sample = np.zeros((int(y.shape[0] * y.shape[1] / step), 1))

    n_sample = 0
    # Take samples along the timeline
    for j in range(0, y.shape[1] - seq_len + 1, step):
        s_index = n_sample
        # Traverse all stock samples
        for i in range(y.shape[0]):
            x1_one = x1[i, j + seq_len - seq_len:j + seq_len]
            y_one = y[i, j + seq_len - 1]

            if (np.isnan(x1_one).any() or (x1_one[-1, :] == 0).any() or np.isnan(
                    y_one).any()):
                continue
            x1_one_last = np.tile(x1_one[-1, :], (x1_one.shape[0], 1))
            x1_one = x1_one / x1_one_last

            x1_in_sample[n_sample, :, :] = x1_one
            y_in_sample[n_sample, :] = y_one
            n_sample += 1
        e_index = n_sample
        if e_index == s_index:
            continue
        y_in_sample[s_index:e_index, 0] = standardize_label(y_in_sample[s_index:e_index, 0])

    x1_in_sample = x1_in_sample[:n_sample, :]
    y_in_sample = y_in_sample[:n_sample, :]

    split = int(y_in_sample.shape[0] * 0.8)
    x1_train = x1_in_sample[:split, :, :]
    x1_val = x1_in_sample[split:, :, :]
    y_train = y_in_sample[:split, :]
    y_val = y_in_sample[split:, :]

    return x1_train, x1_val, y_train, y_val


def load_test_data(x1, y, seq_len=40):

    x1_in_sample = np.zeros(
        (int(x1.shape[0]), seq_len, x1.shape[2]))
    y_in_sample = np.zeros((int(y.shape[0]), 1))
    n_sample = 0
    s_index = n_sample
    nonan_index = []
    for i in range(y.shape[0]):
        x1_one = x1[i, :]
        y_one = y[i, -1]

        if (np.isnan(x1_one).any() or (x1_one[-1, :] == 0).any()):
            continue
        nonan_index.append(i)
        x1_one_last = np.tile(x1_one[-1, :], (x1_one.shape[0], 1))
        x1_one = x1_one / x1_one_last

        x1_in_sample[n_sample, :, :] = x1_one
        y_in_sample[n_sample, :] = y_one
        n_sample += 1

    e_index = n_sample
    y_in_sample[s_index:e_index, 0] = standardize_label(y_in_sample[s_index:e_index, 0])

    x1_in_sample = x1_in_sample[:n_sample, :]
    y_in_sample = y_in_sample[:n_sample, :]

    x1_test = x1_in_sample
    y_test = y_in_sample

    return x1_test, y_test, nonan_index
