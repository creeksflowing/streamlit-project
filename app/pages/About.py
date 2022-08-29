import streamlit as st
# import time
import glob
from pathlib import Path
import acoustics
import acoustics.bands
import acoustics.octave
import numpy as np
from acoustics.bands import (
    _check_band_type, octave_low, octave_high, third_low, third_high)
from acoustics.signal import bandpass
from acoustics.standards.iec_61672_1_2013 import (NOMINAL_OCTAVE_CENTER_FREQUENCIES,
                                                  NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES)
from numba import jit
from scipy import signal
from scipy import stats
# import soundfile as sf
from scipy.io import wavfile

try:
    from pyfftw.interfaces.numpy_fft import rfft
except ImportError:
    from numpy.fft import rfft

OCTAVE_CENTER_FREQUENCIES = NOMINAL_OCTAVE_CENTER_FREQUENCIES
THIRD_OCTAVE_CENTER_FREQUENCIES = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES

sample_rate = 48000
sweep_duration = 15
starting_frequency = 20.0
ending_frequency = 24000.0
number_of_repetitions = 3


@jit(nopython=True)
def fade(data, gain_start,
         gain_end):
    """
    Create a fade on an input object

    Parameters
    ----------
    data : object
       An input object

    gain_start : scalar
        Fade starting point

    gain_end : scalar
        Fade end point

    Returns
    -------
    data : object
        An input object with the fade applied
    """
    gain = gain_start
    delta = (gain_end - gain_start) / (len(data) - 1)
    for i in range(len(data)):
        data[i] = data[i] * gain
        gain = gain + delta

    return data


@jit(nopython=True)
def generate_exponential_sweep(time_in_seconds, sr):
    """
    Generate an exponential sweep using Farina's log sweep theory

    Parameters
    ----------
    time_in_seconds : scalar
       Duration of the sweep in seconds

    sr : scalar
        The sampling frequency

    Returns
    -------
    exponential_sweep : array
        An array with the fade() function applied
    """
    time_in_samples = time_in_seconds * sr  # Calcolo tempo in campioni
    # Inizializzo array di zeri ([0] = quanti slot)
    exponential_sweep = np.zeros(time_in_samples, dtype=np.double)
    for n in range(time_in_samples):
        t = n / sr
        exponential_sweep[n] = np.sin(
            (2.0 * np.pi * starting_frequency * sweep_duration)
            / np.log(ending_frequency / starting_frequency)
            * (np.exp((t / sweep_duration) * np.log(ending_frequency / starting_frequency)) - 1.0))

    number_of_samples = 50
    exponential_sweep[-number_of_samples:] = fade(
        exponential_sweep[-number_of_samples:], 1, 0)
    # Gli ultimi campioni dell'array vengono fadeati
    # a[-2:] last two items in the array

    return exponential_sweep


@jit(nopython=True)
def generate_inverse_filter(time_in_seconds, sr,
                            exponential_sweep):
    """
    Generate an inverse filter using Farina's log sweep theory

    Parameters
    ----------
    time_in_seconds : scalar
        Duration of the sweep in seconds

    sr : scalar
        The sampling frequency

    exponential_sweep : array
        The result of the generate_exponential_sweep() function


    Returns
    -------
    inverse_filter : array
         The array resulting from applying an amplitude envelope to the exponential_sweep array
    """
    time_in_samples = time_in_seconds * sr
    amplitude_envelope = np.zeros(time_in_samples, dtype=np.double)
    inverse_filter = np.zeros(time_in_samples, dtype=np.double)
    for n in range(time_in_samples):
        amplitude_envelope[n] = pow(10, (
            (-6 * np.log2(ending_frequency / starting_frequency)) * (n / time_in_samples)) * 0.05)
        inverse_filter[n] = exponential_sweep[-n] * \
            amplitude_envelope[n]  # -n perché inverso

    return inverse_filter


def deconvolve(ir_sweep, ir_inverse):
    """
    A deconvolution of the exponential sweep and the relative inverse filter

    Parameters
    ----------
    ir_sweep : array
        The result of the generate_exponential_sweep() function

    ir_inverse : array
        The result of the generate_inverse_filter() function

    Returns
    -------
    normalized_ir : array
         An N-dimensional array containing a subset of the discrete linear deconvolution of ir_sweep with ir_inverse
    """
    impulse_response = signal.fftconvolve(ir_sweep, ir_inverse,
                                          mode='full')  # Convolve two N-dimensional arrays using FFT

    # Normalizzo prendendo il picco massimo
    normalized_ir = impulse_response * (1.0 / np.max(abs(impulse_response)))
    # tenerlo normalizzato solo per visualizzare wav
    # e dividendolo per il punto massimo in valore assoluto dell'array

    """
    La deconvoluzione della risposta impulsiva del sistema può essere ottenuta eseguendo una convoluzione (mathematical
    operation on two functions (f and g) that produces a third function (f*g) that expresses how the shape of one is
    modified by the other) del segnale di output registrato y(t) con il suo filtro inverso f(t):
    h(t) = y(t) ⊗ f(t)
    """

    return normalized_ir


def third(first, last):
    """
    Generate a Numpy array for central frequencies of third octave bands.

    Parameters
    ----------
    first : scalar
       First third octave center frequency.

    last : scalar
        Last third octave center frequency.

    Returns
    -------
    octave_bands : array
        An array of center frequency third octave bands.
    """

    """
    la classe ritorna una spettro frequenziale ad ottave, che può essere reso frazionario dal parametro fraction,
    in questo caso reso uguale a 3, ottenendo quindi uno spettro a bande di terzi di ottave. La classe verifica che
    first e last siano frequenze contenute all’interno dello standard IEC 61672_1_2013, se il controllo è positivo
    first viene assegnato come valore di frequenza più basso e last come valore di frequenza più alto dello spettro,
    viene poi creato lo spettro frequenziale includendo tutte le frequenze centrali nominali incluse nello standard
    in base al range definito.
    """
    return acoustics.signal.OctaveBand(fstart=first, fstop=last, fraction=3).nominal


def t60_impulse(file, bands,
                rt='t30'):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.

    Parameters
    ----------
    file: .wav
        Name of the WAV file containing the impulse response.

    bands: array
        Octave or third bands as NumPy array.

    rt: instruction
        Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.

    Returns
    -------
    t60: array
        Reverberation time :math:`T_{60}`
    """
    fs, raw_signal = wavfile.read(file)
    band_type = _check_band_type(bands)

    """
    vengono inizializzate le frequenze di taglio per le basse e le alte frequenze centrali nominali dello spettro
    tramite le funzioni proprietarie della libreria python-acoustics third_low e third_high, che altro non fanno se non
    dividere il risultato della funzione third per 2^(1/6).
    """
    if band_type == 'octave':
        # [-1] = last element in the list
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    # Obbligo rt ad essere lower-case
    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        """
        definisce la frequenza di Nyquist (frequenza di campionamento * 0.5, solo sulle alte) per poi dividere i
        parametri low ed high
        per quest’ultima; viene poi applicata la funzione butter proprietaria della libreria scipy, un filtro
        butterworth, il quale scopo è mantenere il più piatto possibile il modulo della risposta in frequenza nella
        banda passante.
        """
        filtered_signal = bandpass(
            raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        """
        una curva ottenuta eseguendo un'integrazione (ovvero una ricerca delle primitive della funzione, quindi una
        ricerca di una funzione derivabile la cui derivata è uguale alla funzione di partenza) all'indietro della
        risposta all'impulso al quadrato, che idealmente inizia in un punto in cui la risposta decade nel rumore, ed
        applicando ad essa una correzione (il valore di partenza per l'integrazione), che implica il rapporto al quale
        la curva di Schroeder continua a scendere per l'intera risposta. La pendenza di questa curva è usata per
        misurare quanto è veloce il decadimento della risposta all'impulso, derivando una figura per la "RT60",
        che è il tempo che ci impiega il suono a scendere di livello di 60dB.
        """
        # np.cumsum, utilizzata per visualizzare la somma totale dei dati man mano che crescono nel tempo
        sch = np.cumsum(abs_signal[::-1] ** 2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))

        # Linear regression
        # indice minimo inizio
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]  # indice minimo fine
        init_sample = np.where(sch_db == sch_init)[0][0]  # dove inizia indice
        end_sample = np.where(sch_db == sch_end)[0][0]  # dove inizia indice
        # trovo in secondi il decadimento #arange ritorna valori
        x = np.arange(init_sample, end_sample + 1) / fs
        # equalmente spaziati tra gli intervalli dati
        # tutto l'array tranne il campione iniziale e quello finale
        y = sch_db[init_sample:end_sample + 1]
        # calcolo la regressione lineare, estraiamo i parametri di slope
        slope, intercept = stats.linregress(x, y)[0:2]
        # ed intercept (Intercept of the regression line), parametri utili al calcolo dei valori di inizio e fine
        # regressione per trovare il T60

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)

    return t60


def sabine_absorption_area(sabine_t60, volume):
    """
    Equivalent absorption surface or area in square meters following Sabine's formula

    Parameters
    ----------
    sabine_t60: array
        The result of the t60_impulse() function

    volume: scalar
        The volume of the room

    Returns
    -------
    absorption_area: scalar
        The equivalent absorption surface
    """
    absorption_area = (0.161 * volume) / sabine_t60

    return absorption_area


for n in range(1, number_of_repetitions + 1):
    sweep = generate_exponential_sweep(sweep_duration, sample_rate)
    inverse = generate_inverse_filter(sweep_duration, sample_rate, sweep)
    ir = deconvolve(sweep, inverse)

    sweep_name = "exponential_sweep_%d.wav" % (n,)
    inverse_filter_name = "inverse_filter_%d.wav" % (n,)
    ir_name = "impulse_response_%d.wav" % (n,)

    wavfile.write(f"{sweep_name}", sample_rate, sweep)
    wavfile.write(f"{inverse_filter_name}", sample_rate, inverse)
    wavfile.write(f"{ir_name}", sample_rate, ir)
"""
    if n != number_of_repetitions:
        time.sleep(sweep_duration)
"""

ir_list = []
for file_name in glob.glob("impulse_response_*.*"):
    sample_rate, data = wavfile.read(file_name)
    ir_list.append(data)
mean_ir = np.mean(ir_list)
wavfile.write("mean_ir.wav", sample_rate, ir)
datadir = Path("/Users/ettorecarlessi/Documents/PyCharm/Projects/rev_room")

file_ir = datadir / "mean_ir.wav"
f = open(file_ir)
t60 = t60_impulse(file_ir, third(100, 5000), rt='t30')
print(t60)
print(sabine_absorption_area(t60, 300))

# jna volta avuto il t60 calcolo assorbimento equivalente.
