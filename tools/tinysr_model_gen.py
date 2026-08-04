#!/usr/bin/env python3
"""Python 3 port of tinysr/scripts/model_gen.py, plus held-out evaluation.

Usage:
    tinysr_model_gen.py data/cat data/dog [...] output_model
    tinysr_model_gen.py --eval-holdout 2 data/cat data/dog [...] output_model

Each directory holds utterance CSVs produced by tinysr/apps/store_utters.
With --eval-holdout N, N utterances per word are held out of training and
classified afterwards; a confusion matrix and accuracy are printed. The model
written to output_model is trained on ALL utterances (eval is informational).
"""
import argparse
import math
import os
import random
import struct
import sys
import time

import numpy


class MultivariateGaussianModel:
    def __init__(self, vecs):
        self.mean = sum(vecs) / len(vecs)
        vecs = [v - self.mean for v in vecs]
        if len(vecs) == 1:
            self.covariance = numpy.identity(len(vecs[0]))
        else:
            self.covariance = sum(numpy.outer(v, v) for v in vecs) / len(vecs)
        # regularize: store_utters data can make near-singular covariances
        self.covariance += numpy.identity(len(self.mean)) * 1e-6
        self.covar_inv = numpy.linalg.inv(self.covariance)
        sign, logdet = numpy.linalg.slogdet(self.covariance)
        self.ll_const = -0.5 * logdet

    def ll(self, datum):
        datum = numpy.asarray(datum) - self.mean
        return self.ll_const - 0.5 * datum.dot(self.covar_inv.dot(datum))


class Model:
    def __init__(self, name, stacks):
        self.name, self.stacks = name, stacks
        self.ll_offset, self.ll_slope = 0.0, 1.0

    def build_model(self):
        self.template = [MultivariateGaussianModel(stack) for stack in self.stacks]

    def dynamic_time_warping(self, utterance):
        ll = {}
        parent = {}

        def get(xy):
            return ll.get(xy, -float("inf"))

        for x in range(len(utterance)):
            for y in range(len(self.template)):
                min_predec = max([(x - 1, y), (x, y - 1), (x - 1, y - 1)], key=get)
                ll[x, y] = (0.0 if (x, y) == (0, 0) else ll[min_predec]) + self.template[y].ll(utterance[x])
                parent[x, y] = min_predec
        parent[0, 0] = None
        path = [(len(utterance) - 1, len(self.template) - 1)]
        while path[0] in parent:
            path.insert(0, parent[path[0]])
        return ll[path[-1]], path[1:]

    def ll(self, utterance):
        return self.dynamic_time_warping(utterance)[0]

    def normalized_ll(self, utterance):
        return self.ll_offset + self.ll_slope * self.ll(utterance)

    def write_to_file(self, f):
        name_bytes = self.name.encode("utf-8")
        f.write(struct.pack("<I", len(name_bytes)) + name_bytes)
        f.write(struct.pack("<2f", self.ll_offset, self.ll_slope))
        f.write(struct.pack("<I", len(self.template)))
        for gaussian in self.template:
            f.write(struct.pack("<f", gaussian.ll_const))
            f.write(struct.pack("<13f", *gaussian.mean))
            for row in gaussian.covar_inv:
                f.write(struct.pack("<13f", *row))


def read_in_utterance(path):
    fvs = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            fvs.append(numpy.array([float(v) for v in line.split(",")][1:]))
    assert len(fvs[0]) == 13, f"{path}: expected 13 cepstral coefficients"
    return fvs


def load_dir(utters_path):
    utters = []
    for name in sorted(os.listdir(utters_path)):
        if name.endswith(".csv"):
            utters.append(read_in_utterance(os.path.join(utters_path, name)))
    return utters


def build_model_from_utters(name, utters):
    l = sorted(utters, key=len)
    candidate = l[len(l) // 2]
    print(f"  utters={len(utters)} lengths: min={len(l[0])} median={len(candidate)} max={len(l[-1])}")
    model = Model(name, [[fv] for fv in candidate])
    model.build_model()

    old_error, round_num = float("-inf"), 0
    while True:
        round_num += 1
        new_model = Model(name, [[] for _ in range(len(model.template))])
        total_fit_error = 0.0
        for u in utters:
            ll, path = model.dynamic_time_warping(u)
            total_fit_error += ll
            for x, y in path:
                new_model.stacks[y].append(u[x])
        new_model.build_model()
        print(f"  round {round_num} log likelihood: {total_fit_error:.1f}")
        if total_fit_error <= old_error:
            break
        old_error = total_fit_error
        model = new_model
    return model


def cross_normalize(models):
    """models: list of (utters, Model, name)."""
    for utters, model, word_name in models:
        match_ll = sum(model.ll(u) for u in utters) / len(utters)
        reject_utters = [u for us, m, n in models if m is not model for u in us]
        reject_ll = sum(model.ll(u) for u in reject_utters) / len(reject_utters)
        denom = reject_ll - match_ll
        if abs(denom) < 1e-9:
            denom = -1.0
        slope = -100.0 / denom
        offset = -match_ll * slope
        print(f"  {word_name}: match={match_ll:.1f} reject={reject_ll:.1f} -> new = {offset:.2f} + {slope:.4f} * old")
        model.ll_offset = offset
        model.ll_slope = slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="word dirs... output_model")
    ap.add_argument("--eval-holdout", type=int, default=0, help="held-out utterances per word for accuracy eval")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if len(args.paths) < 2:
        ap.error("need at least one word directory and an output path")
    input_paths = args.paths[:-1]
    output_path = args.paths[-1]
    rng = random.Random(args.seed)

    words = []
    for path in input_paths:
        name = os.path.basename(os.path.normpath(path))
        utters = load_dir(path)
        if not utters:
            sys.exit(f"no utterance CSVs in {path}")
        words.append((name, utters))

    # optional held-out evaluation
    if args.eval_holdout > 0:
        print(f"=== Held-out evaluation ({args.eval_holdout} per word) ===")
        train_models = []
        test_sets = []
        for name, utters in words:
            if len(utters) <= args.eval_holdout:
                sys.exit(f"word {name}: need more than {args.eval_holdout} utterances (have {len(utters)})")
            shuffled = utters[:]
            rng.shuffle(shuffled)
            test = shuffled[: args.eval_holdout]
            train = shuffled[args.eval_holdout:]
            print(f"== training eval model {name!r} on {len(train)}")
            model = build_model_from_utters(name, train)
            train_models.append((train, model, name))
            test_sets.append((name, test))
        cross_normalize(train_models)

        names = [n for _, _, n in train_models]
        confusion = {n: {m: 0 for m in names} for n in names}
        correct = total = 0
        for true_name, test in test_sets:
            for u in test:
                scores = [(m.normalized_ll(u), n) for _, m, n in train_models]
                best_score, best_name = max(scores)
                confusion[true_name][best_name] += 1
                total += 1
                correct += int(best_name == true_name)
                marker = "OK " if best_name == true_name else "MISS"
                print(f"  [{marker}] true={true_name:10s} -> {best_name:10s} (score {best_score:.1f})")
        print("\nConfusion (rows=true, cols=predicted):")
        header = " " * 11 + " ".join(f"{n[:8]:>8s}" for n in names)
        print(header)
        for n in names:
            print(f"{n[:10]:10s} " + " ".join(f"{confusion[n][m]:8d}" for m in names))
        print(f"\nACCURACY: {correct}/{total} = {100.0 * correct / total:.1f}%\n")

    # final model on all data
    print(f"=== Building final model with {len(words)} words ===")
    models = []
    for name, utters in words:
        print(f"== {name!r}")
        model = build_model_from_utters(name, utters)
        models.append((utters, model, name))
    print("=== Cross normalizing ===")
    cross_normalize(models)
    with open(output_path, "wb") as f:
        for utters, model, word_name in models:
            model.write_to_file(f)
    print(f"wrote {output_path} ({os.path.getsize(output_path):,} bytes)")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"done in {time.time() - start:.1f}s")
