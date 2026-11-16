"""Dataset loading and preprocessing for NSL-KDD, UNSW-NB15 and TON_IoT (network)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import os
D = os.environ.get("CHIDS_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")) + "/"

NSL_COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
            "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
            "attack", "difficulty"]

NSL_MAP = {}
for a in ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb",
          "processtable", "udpstorm", "worm"]:
    NSL_MAP[a] = "DoS"
for a in ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"]:
    NSL_MAP[a] = "Probe"
for a in ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient",
          "warezmaster", "sendmail", "named", "snmpgetattack", "snmpguess", "xlock", "xsnoop",
          "httptunnel"]:
    NSL_MAP[a] = "R2L"
for a in ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"]:
    NSL_MAP[a] = "U2R"
NSL_MAP["normal"] = "Normal"


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor(num_cols, cat_cols):
    num = Pipeline([("slog", FunctionTransformer(_slog, feature_names_out="one-to-one")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                        sparse_output=False)
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def load_nsl():
    tr = pd.read_csv(D + "KDDTrain+.txt", header=None, names=NSL_COLS)
    te = pd.read_csv(D + "KDDTest+.txt", header=None, names=NSL_COLS)
    te21 = pd.read_csv(D + "KDDTest-21.txt", header=None, names=NSL_COLS)
    out = []
    for df in (tr, te, te21):
        fam = df["attack"].map(NSL_MAP)
        assert fam.notna().all(), df["attack"][fam.isna()].unique()
        y = (fam != "Normal").astype(int).values
        X = df.drop(columns=["attack", "difficulty", "num_outbound_cmds"])
        out.append((X, y, fam.values, df["attack"].values))
    cat = ["protocol_type", "service", "flag"]
    num = [c for c in out[0][0].columns if c not in cat]
    train_attacks = set(out[0][3])
    return dict(name="NSL-KDD", train=out[0], tests={"KDDTest+": out[1], "KDDTest-21": out[2]},
                num=num, cat=cat, train_attacks=train_attacks)


def load_unsw():
    # official split: training set 175,341 records, testing set 82,332 records
    # (the file names in the redistributed copy are swapped, so we assign by size)
    a = pd.read_csv(D + "UNSW_NB15_testing-set.csv")
    b = pd.read_csv(D + "UNSW_NB15_training-set.csv")
    tr, te = (a, b) if len(a) > len(b) else (b, a)
    assert len(tr) == 175341 and len(te) == 82332
    out = []
    for df in (tr, te):
        fam = df["attack_cat"].str.strip().values
        y = df["label"].values.astype(int)
        X = df.drop(columns=["id", "attack_cat", "label"])
        out.append((X, y, fam, fam))
    cat = ["proto", "service", "state"]
    num = [c for c in out[0][0].columns if c not in cat]
    return dict(name="UNSW-NB15", train=out[0], tests={"UNSW-NB15 test": out[1]}, num=num, cat=cat)


def load_ton(seed=42):
    df = pd.read_csv(D + "train_test_network.csv", low_memory=False)
    df.columns = [c.replace("﻿", "") for c in df.columns]
    fam = df["type"].str.strip().str.capitalize().replace({"Ddos": "DDoS", "Dos": "DoS",
                                                           "Xss": "XSS", "Mitm": "MITM"}).values
    y = df["label"].values.astype(int)
    # drop identifiers (IPs, ports) and free-text, high-cardinality fields to avoid shortcut learning
    drop = ["src_ip", "dst_ip", "src_port", "dst_port", "dns_query", "ssl_subject", "ssl_issuer",
            "http_uri", "http_user_agent", "weird_name", "weird_addl", "label", "type"]
    X = df.drop(columns=drop)
    cat = [c for c in X.columns if X[c].dtype == object]
    for c in cat:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cat]
    X[num] = X[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    idx = np.arange(len(df))
    itr, ite = train_test_split(idx, test_size=0.15, stratify=fam, random_state=seed)
    mk = lambda ii: (X.iloc[ii].reset_index(drop=True), y[ii], fam[ii], fam[ii])
    # note: with pandas' string dtype the object-dtype test above selects no column, so all string fields are
    # coerced to 0 and the model effectively uses the numeric fields only (as described in the paper)
    return dict(name="TON_IoT", train=mk(itr), tests={"TON_IoT test": mk(ite)}, num=num, cat=cat,
                raw_proto_train=df["proto"].astype(str).values[itr])


LOADERS = {"NSL-KDD": load_nsl, "UNSW-NB15": load_unsw, "TON_IoT": load_ton}


def prepare(ds, seed, cal_frac=0.15 / 0.85):
    """Split training data into fit / calibration (stratified by family) and transform.
    Returns dense float32 matrices."""
    X, y, fam, sub = ds["train"]
    idx = np.arange(len(y))
    ifit, ical = train_test_split(idx, test_size=cal_frac, stratify=fam, random_state=seed)
    pre = make_preprocessor(ds["num"], ds["cat"])
    Xf = pre.fit_transform(X.iloc[ifit]).astype(np.float32)
    Xc = pre.transform(X.iloc[ical]).astype(np.float32)
    tests = {}
    for k, (Xt, yt, ft, st) in ds["tests"].items():
        tests[k] = (pre.transform(Xt).astype(np.float32), yt, ft, st)
    names = list(pre.get_feature_names_out())
    return dict(Xf=Xf, yf=y[ifit], ff=fam[ifit], Xc=Xc, yc=y[ical], fc=fam[ical], tests=tests,
                names=names, pre=pre, ifit=ifit, ical=ical)
