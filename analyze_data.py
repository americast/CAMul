import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pudb
np.random.seed(10)
from tqdm import tqdm
from optparse import OptionParser
from models.utils import float_tensor, device

regions = ["CA", "FL", "MA", "NY"]
city_idx = {r: i for i, r in enumerate(regions)}

features = [
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
    "covidnet",
    "positiveIncrease",
    "negativeIncrease",
    "totalTestResultsIncrease",
    "onVentilatorCurrently",
    "inIcuCurrently",
    "recovered",
    "hospitalizedIncrease",
    "death_jhu_incidence",
    "dex_a",
    "apple_mobility",
    "Number of Facilities Reporting",
    "CLI Percent of Total Visits",
]

label = "death_jhu_incidence"

full_seqs = []
for r in tqdm(regions):
    with open(f"./data/covid_data/saves/backfill_vals_{r}.pkl", "rb") as fl:
        data = pickle.load(fl)
    full_seq = [[data[0][feat][i][0] for i in data[0][feat]] for feat in features]
    # full_seq = [full_seq[i][-1] for i in full_seq]
    full_seqs.append(full_seq)
    os.system("mkdir -p plots_orig_data/"+r)
    for f, feat in enumerate(features):
        to_plot = full_seq[f]
        plt.plot(to_plot)
        plt.savefig("plots_orig_data/"+r+"/"+feat+".png")
        plt.clf()
full_seqs = np.array(full_seqs).transpose(0, 2, 1)  # Shape: [regions, time, features]

# Normalize
# One scaler per region
scalers = [StandardScaler() for _ in range(len(regions))]
full_seqs_norm = []
for i, scaler in enumerate(scalers):
    full_seqs_norm.append(scaler.fit_transform(full_seqs[i]))
# pu.db
for i, r in enumerate(tqdm(regions)):
    os.system("mkdir -p plots_normalized_data/"+r)
    full_seq = full_seqs_norm[i]
    for f, feat in enumerate(features):
        to_plot = full_seq[:, f]
        plt.plot(to_plot)
        plt.savefig("plots_normalized_data/"+r+"/"+feat+".png")
        plt.clf()

full_seqs_norm = np.array(full_seqs_norm)

parser = OptionParser()
parser.add_option("-w", "--week", dest="week_ahead", type="int", default=2)
parser.add_option("-a", "--atten", dest="atten", type="string", default="trans")
parser.add_option("-n", "--num", dest="num", type="string")
parser.add_option("-e", "--epoch", dest="epochs", type="int", default="1")
parser.add_option("-b", "--break", dest="break_ref_set", type="int", default=4)

(options, args) = parser.parse_args()

BREAK_REF_SET = options.break_ref_set
print("Break ref sets into "+str(BREAK_REF_SET)+" part(s).")

# train_seasons = list(range(2003, 2019))
# test_seasons = [2019]

# train_seasons = [2003, 2004, 2005, 2006, 2007, 2008, 2009]
# test_seasons = [2010]
# regions = ["X"]
# regions = [f"Region {i}" for i in range(1,11)]

week_ahead = options.week_ahead
val_frac = 5
attn = options.atten
model_num = options.num
# model_num = 22
EPOCHS = options.epochs
print(week_ahead, attn, EPOCHS)


def one_hot(idx, dim=len(city_idx)):
    ans = np.zeros(dim, dtype="float32")
    ans[idx] = 1.0
    return ans


def save_data(obj, filepath):
    with open(filepath, "wb") as fl:
        pickle.dump(obj, fl)


full_x_all = np.array(full_seqs_norm)

full_meta_all = np.array([one_hot(city_idx[r]) for r in regions])
full_y_all = full_x_all[:, :, features.index(label)].argmax(-1)
# pu.db
#full_test_x = full_x[:, -8:, :]
#full_test_meta = full_meta
#full_test_y = full_y
# full_x = full_x[:, :-8, :]


def create_dataset(full_meta, full_x, week_ahead=week_ahead):
    metas, seqs, y = [], [], []
    for meta, seq in zip(full_meta, full_x):
        for i in range(5, full_x.shape[1]):
            metas.append(meta)
            seqs.append(seq[: i - week_ahead + 1])
            y.append(seq[i, features.index(label), None])
    return np.array(metas, dtype="float32"), seqs, np.array(y, dtype="float32")


def create_dataset2(full_meta, full_x, week_ahead=week_ahead, last=5):
    metas, seqs, y = [], [], []
    metas_t, seqs_t, y_t = [], [], []
    for meta, seq in zip(full_meta, full_x):
        for i in range(5, full_x.shape[1] - last):
            metas.append(meta)
            seqs.append(seq[: i - week_ahead + 1])
            y.append(seq[i, features.index(label), None])
        for i in range(full_x.shape[1] - last-10, full_x.shape[1]):
            metas_t.append(meta)
            seqs_t.append(seq[: i - week_ahead + 1])
            y_t.append(seq[i, features.index(label), None])
    return (
        np.array(metas, dtype="float32"),
        seqs,
        np.array(y, dtype="float32"),
        np.array(metas_t, dtype="float32"),
        seqs_t,
        np.array(y_t, dtype="float32"),
    )

for w in range(4,5):

    full_x, full_meta = full_x_all[:,:-w,:], full_meta_all
    full_y = full_x[:, :, features.index(label)].argmax(-1)
    # train_meta, train_x, train_y = create_dataset(full_meta, full_x)
    # test_meta, test_x, test_y = train_meta, train_x, train_y
    train_meta, train_x, train_y, test_meta, test_x, test_y = create_dataset2(
        full_meta_all, full_x_all, last=week_ahead
    )
    os.system("mkdir -p plots_test_x/")
    for k, feat in enumerate(features):
        to_plot = test_x[0][:, k]
        plt.plot(to_plot)
        plt.savefig("plots_test_x/"+feat+".png")
        plt.clf()

    def create_tensors(metas, seqs, ys):
        metas = float_tensor(metas)
        ys = float_tensor(ys)
        max_len = max([len(s) for s in seqs])
        out_seqs = np.zeros((len(seqs), max_len, seqs[0].shape[-1]), dtype="float32")
        lens = np.zeros(len(seqs), dtype="int32")
        for i, s in enumerate(seqs):
            out_seqs[i, : len(s), :] = s
            lens[i] = len(s)
        out_seqs = float_tensor(out_seqs)
        return metas, out_seqs, ys, lens


    def create_mask1(lens, out_dim=1):
        ans = np.zeros((max(lens), len(lens), out_dim), dtype="float32")
        for i, j in enumerate(lens):
            ans[j - 1, i, :] = 1.0
        return float_tensor(ans)


    def create_mask(lens, out_dim=1):
        ans = np.zeros((max(lens), len(lens), out_dim), dtype="float32")
        for i, j in enumerate(lens):
            ans[:j, i, :] = 1.0
        return float_tensor(ans)


    # if attn == "trans":
    #     emb_model = EmbedAttenSeq(
    #         dim_seq_in=len(features),
    #         dim_metadata=len(city_idx),
    #         dim_out=50,
    #         n_layers=2,
    #         bidirectional=True,
    #     ).cuda()
    #     emb_model_full = EmbedAttenSeq(
    #         dim_seq_in=len(features),
    #         dim_metadata=len(city_idx),
    #         dim_out=50,
    #         n_layers=2,
    #         bidirectional=True,
    #     ).cuda()
    # else:
    #     emb_model = EmbedSeq(
    #         dim_seq_in=len(features),
    #         dim_metadata=len(city_idx),
    #         dim_out=50,
    #         n_layers=2,
    #         bidirectional=True,
    #     ).cuda()
    #     emb_model_full = EmbedSeq(
    #         dim_seq_in=len(features),
    #         dim_metadata=len(city_idx),
    #         dim_out=50,
    #         n_layers=2,
    #         bidirectional=True,
    #     ).cuda()
    # fnp_model = RegressionFNP2(
    #     dim_x=50,
    #     dim_y=1,
    #     dim_h=100,
    #     n_layers=3,
    #     num_M=train_meta.shape[0],
    #     dim_u=50,
    #     dim_z=50,
    #     fb_z=0.0,
    #     use_ref_labels=False,
    #     use_DAG=False,
    #     add_atten=False,
    # ).cuda()
    # optimizer = optim.Adam(
    #     list(emb_model.parameters())
    #     + list(fnp_model.parameters())
    #     + list(emb_model_full.parameters()),
    #     lr=1e-3,
    # )

    # emb_model_full = emb_model

    train_meta_, train_x_, train_y_, train_lens_ = create_tensors(
        train_meta, train_x, train_y
    )

    test_meta, test_x, test_y, test_lens = create_tensors(test_meta, test_x, test_y)
    full_x_chunks = np.zeros((full_x.shape[0] * 4, full_x.shape[1], full_x.shape[2]))
    full_meta_chunks = np.zeros((full_meta.shape[0] * 4, full_meta.shape[1]))
    for i, s in enumerate(full_x):
        full_x_chunks[i * 4, -20:] = s[:20]
        full_x_chunks[i * 4 + 1, -30:] = s[:30]
        full_x_chunks[i * 4 + 2, -40:] = s[:40]
        full_x_chunks[i * 4 + 3, :] = s
        full_meta_chunks[i * 4 : i * 4 + 4] = full_meta[i]
    # BREAK_REF_SET = 5
    os.system("mkdir -p plots_broken_into_"+str(BREAK_REF_SET)+"_data/")
    ilk = full_x.shape[1]//BREAK_REF_SET
    # pu.db
    to_concat = []
    for bn in range(BREAK_REF_SET):
        to_concat.append(full_x[:,bn*ilk:(bn+1)*ilk,:])
    if BREAK_REF_SET != 4:
        full_x = np.concatenate(to_concat)
        full_meta = np.concatenate([full_meta for _ in range(BREAK_REF_SET)])
    elif BREAK_REF_SET == 5:
        full_x = np.concatenate([full_x[:,:ilk,:], full_x[:,ilk:2*ilk,:], full_x[:,2*ilk:3*ilk,:],full_x[:,3*ilk:4*ilk,:], full_x[:,4*ilk:5*ilk,:]])
        full_meta = np.concatenate([full_meta for _ in range(5)])
    elif BREAK_REF_SET == 3:
        full_x = np.concatenate([full_x[:,:ilk,:], full_x[:,ilk:2*ilk,:], full_x[:,2*ilk:3*ilk,:]])
        full_meta = np.concatenate([full_meta for _ in range(3)])
    elif BREAK_REF_SET == 4:
        full_x = np.concatenate([full_x[:,:ilk,:], full_x[:,ilk:2*ilk,:], full_x[:,2*ilk:3*ilk,:],full_x[:,3*ilk:4*ilk,:]])
        full_meta = np.concatenate([full_meta for _ in range(4)])
    elif BREAK_REF_SET == 6:
        full_x = np.concatenate([full_x[:,:ilk,:], full_x[:,ilk:2*ilk,:], full_x[:,2*ilk:3*ilk,:],full_x[:,3*ilk:4*ilk,:], full_x[:,4*ilk:5*ilk,:], full_x[:,5*ilk:6*ilk,:]])
        full_meta = np.concatenate([full_meta for _ in range(6)])
    
    for i, each_concat in enumerate(tqdm(to_concat)):
        os.system("mkdir -p plots_broken_into_"+str(BREAK_REF_SET)+"_data/part_"+str(i+1))
        for j, seq in enumerate(each_concat):
            r = regions[j]
            os.system("mkdir -p plots_broken_into_"+str(BREAK_REF_SET)+"_data/part_"+str(i+1)+"/"+r)
            for k, feat in enumerate(features):
                to_plot = seq[:, k]
                plt.plot(to_plot)
                plt.savefig("plots_broken_into_"+str(BREAK_REF_SET)+"_data/part_"+str(i+1)+"/"+r+"/"+feat+".png")
                plt.clf()