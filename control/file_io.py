import csv
import os
from pathlib import Path
import pymoto as pym
import numpy as np
import pickle


class Logger(pym.Module):
    def _prepare(self, saveto="out/log.csv", open_viewer=False, iteration=0):
        self.saveto = saveto
        self.iteration = iteration
        self.created = False
        self.lens = None

    def _response(self, *args):
        if not self.created:
            self.lens = [0 for _ in self.sig_in]
            # Find correct field names
            fields = ['Iter']
            for i, s in enumerate(self.sig_in):
                try:
                    n = len(s.state)
                except:
                    n = 1
                self.lens[i] = n
                if n == 1:
                    fields.append(s.tag)
                else:
                    for j in range(n):
                        fields.append(s.tag + " {0:d}".format(j+1))

            self.N = len(fields)

            ## Initialize new file
            with open(self.saveto, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
            self.created = True

        # Gather values
        values = [int(self.iteration)]
        for i, s in enumerate(self.sig_in):
            try:
                for j in range(len(s.state)):
                    values.append(s.state[j])
            except:
                values.append(s.state)

        if len(values) != self.N:
            raise RuntimeError("Not enough values e.g. length of vector has changed")
        if isinstance(self.iteration, int):
            self.iteration += 1
        with open(self.saveto, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(values)


class SaveToNumpy(pym.Module):
    def _prepare(self, saveto="out/data.npy", iter=0):
        if saveto is not None:
            Path(saveto).parent.mkdir(parents=True, exist_ok=True)
            self.saveloc, self.saveext = os.path.splitext(saveto)
        else:
            self.saveloc, self.saveext = None, None
        self.iter = iter

    def _response(self, *args):
        with open("{0:s}.{1:04d}{2:s}".format(self.saveloc, self.iter, self.saveext), 'wb') as f:
            for a in args:
                np.save(f, a)
        if isinstance(self.iter, int):
            self.iter += 1


def crawl_data(dat):
    # Crawls data for saving to pickle
    out = None
    if isinstance(dat, list):
        out = [crawl_data(v) for v in dat]
    elif isinstance(dat, dict):
        out = {}
        for k, v in dat.items():
            out[k] = crawl_data(v)
    elif hasattr(dat, 'state') and hasattr(dat, 'tag'):
        out = (dat.tag, dat.state)
    else:
        print(f"Warning [crawl_data()], Unrecognized data {dat} -- skipping")
    return out


class SaveToPickle(pym.Module):
    def _prepare(self, savedat, saveto="out/data.p", iter=0):
        if saveto is not None:
            Path(saveto).parent.mkdir(parents=True, exist_ok=True)
            self.saveloc, self.saveext = os.path.splitext(saveto)
        else:
            self.saveloc, self.saveext = None, None
        self.iter = iter
        self.savedat = savedat

    def _response(self):
        # Crawl savedat
        with open("{0:s}.{1:04d}{2:s}".format(self.saveloc, self.iter, self.saveext), 'wb') as f:
            pickle.dump(crawl_data(self.savedat), f)

        if isinstance(self.iter, int):
            self.iter += 1