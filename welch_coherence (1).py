import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import welch
from scipy.fft import fftfreq, rfftfreq, fft, rfft
from scipy import signal
import matplotlib.pyplot as plt
from timeflux.core.node import Node


class Coherence_Welch(Node):


    def __init__(self, rate=None, closed="right", **kwargs):
        """
        Args:
            rate (float|None): Nominal sampling rate of the input data. If `None`, the rate will be taken from the input meta/
            closed (str): Make the index closed on the `right`, `left` or `center`.
            kwargs:  Keyword arguments to pass to scipy.signal.welch function.
                            You can specify: window, nperseg, noverlap, nfft, detrend, return_onesided and scaling.
        """

        self._rate = rate
        self._closed = closed
        self._kwargs = kwargs
        self._set_default()

    def _set_default(self):
        # We set the default params if they are not specifies in kwargs in order to check that they are valid, in respect of the length and sampling of the input data.
        if "nperseg" not in self._kwargs.keys():
            self._kwargs["nperseg"] = 256
            self.logger.debug("nperseg := 256")
        if "nfft" not in self._kwargs.keys():
            self._kwargs["nfft"] = self._kwargs["nperseg"]
            self.logger.debug(
                "nfft := nperseg := {nperseg}".format(nperseg=self._kwargs["nperseg"])
            )
        if "noverlap" not in self._kwargs.keys():
            self._kwargs["noverlap"] = self._kwargs["nperseg"] // 2
            self.logger.debug(
                "noverlap := nperseg/2 := {noverlap}".format(
                    noverlap=self._kwargs["noverlap"]
                )
            )

    def _check_nfft(self):
        # Check validity of nfft at first chun
        if not all(
            i <= len(self.i.data)
            for i in [self._kwargs[k] for k in ["nfft", "nperseg", "noverlap"]]
        ):
            raise ValueError(
                "nfft, noverlap and nperseg must be greater than or equal to length of chunk."
            )
        else:
            self._kwargs.update(
                {
                    keyword: int(self._kwargs[keyword])
                    for keyword in ["nfft", "nperseg", "noverlap"]
                }
            )

    def update(self):
        # copy the meta
        self.o = self.i

        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return

        # Check rate
        if self._rate:
            rate = self._rate
        elif "rate" in self.i.meta:
            rate = self.i.meta["rate"]
        else:
            raise ValueError(
                "The rate was neither explicitely defined nor found in the stream meta."
            )

        # At this point, we are sure that we have some data to process
        # apply welch on the data:
        self._check_nfft()
        data=self.i.data.to_numpy()
        print("the shape of self.data",self.i.data.to_numpy().shape)

        elec1_MI = []
        for elec1 in range(data.shape[1]):
          elec2_MI = []
          for elec2 in range(data.shape[1]):
            f, Cxy_MI = signal.coherence(data[:,elec1], data[:,elec2], fs=rate, window='hann',  **self._kwargs)
            elec2_MI.append(Cxy_MI)
          elec1_MI.append(elec2_MI)
        



        NS = np.array(elec1_MI).mean(0)
        NS=NS.transpose(1,0)
        print("NS",NS.shape)
        if self._closed == "left":
            time = self.i.data.index[-1]
        elif self._closed == "center":

            def middle(a):
                return int(np.ceil(len(a) / 2)) - 1

            time = self.i.data.index[middle(self.i.data)]
        else:  # right
            time = self.i.data.index[-1]
        # f is the frequency axis and Pxx the average power of shape (Nfreqs x Nchanels)
        # we reshape Pxx to fit the ('time' x 'freq' x 'space') dimensions
        self.o.data = xr.DataArray(
            np.stack([NS], 0),
            coords=[[time], f, self.i.data.columns],
            dims=["time", "frequency", "space"],
        )