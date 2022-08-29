import os
import streamlit as st
import streamlit.components.v1 as components
import pygame
import pyaudio
import wave
import sys
import boto3
from google.cloud import storage
import sounddevice as sd
import pandas as pd
import numpy as np
import altair as alt
import time
import asyncio
from datetime import datetime
import streamlit as st
import glob
from pathlib import Path
import acoustics
import acoustics.bands
import acoustics.octave
import numpy as np
from acoustics.bands import (
    _check_band_type, octave_low, octave_high, third_low, third_high)
from acoustics.signal import bandpass
from acoustics.standards.iec_61672_1_2013 import (
    NOMINAL_OCTAVE_CENTER_FREQUENCIES,
    NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES
)
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

# Global variables
running = False
hours, minutes, seconds = 0, 0, 0
sample_rate = 48000
sweep_duration = 15
starting_frequency = 20.0
ending_frequency = 24000.0
number_of_repetitions = 1
# GCS_CLIENT = storage.Client()


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


"""
ir_list = []
for file_name in glob.glob("impulse_response_*.*"):
    sample_rate, data = wavfile.read(file_name)
    ir_list.append(data)
    mean_ir = np.mean(ir_list)
    wavfile.write("mean_ir.wav", sample_rate, ir)
    """
"""
    if n != number_of_repetitions:
        time.sleep(sweep_duration)
"""


datadir = Path("/Users/ettorecarlessi/Documents/PyCharm/Projects/rev_room")

file_ir = datadir / "mean_ir.wav"
f = open(file_ir)
t60 = t60_impulse(file_ir, third(100, 5000), rt='t30')
print(t60)
print(sabine_absorption_area(t60, 300))

# jna volta avuto il t60 calcolo assorbimento equivalente.
"""
x = np.arange(100)
        source = pd.DataFrame({
            's': sweep_duration,
            'Amp': np.sin(x / 5)
        })

        c = alt.Chart(source).mark_line().encode(
            x='s',
            y='Amp'
        )

        st.altair_chart(c, use_container_width=True)
"""


def head():
    st.title("Wave(r)")

    st.caption("by Ettore Carlessi")

    st.subheader(
        "Acoustic test and measurement website"
    )


def head_impulse_response():
    st.title("Impulse Response")

    st.caption("Measure your room")

    st.markdown("The **Impulse Response Module** provides an easy way to capture an IR audio file, and also calculates the most-needed metrics from the data acquired.")
    st.markdown("In seconds, a complete set of measurements is made that describes the acoustic character of a room, or the response of a loudspeaker.")
    st.markdown("The Impulse Response Module will give you very good results, even with the built-in mic. Of course, you'll be limited to moderate SPL levels, and the lowest frequency bands will not be as accurate, but this module will give you accurate reverb decay times and other measurements.")


async def time_convert():
    container = st.empty()
    container_2 = st.empty()
    global button_start
    global button_end
    button_start = container_2.button('Start')
    clock = f"{0:02d}:{0:02d}"

    if button_start:
        button_end = container_2.button('End')

        for secs in range(0, 1000, 1):
            mm, ss = secs // 60, secs % 60
            container.metric("Time Lapsed", f"{mm:02d}:{ss:02d}")
            r = await asyncio.sleep(1)

        if button_end:
            container_2.empty()
            button_start = container_2.button('Start')

    else:
        container.metric("Time Lapsed", clock)


def select_sweep_time():
    sweep_duration_option = st.selectbox('Select the duration of the sweep',
                                         ('3s', '7s', '14s'))
    max_reverb_option = st.selectbox('Select the expected maximum reverb decay time',
                                     ('1s', '2s', '3s', '5s', '10s'))
    st.caption('''
        Note that longer sweeps provide more accuacy,
        but even short sweeps can be used to measure long decays
        ''')

    if sweep_duration_option == '3s':
        sweep_duration = 3
    elif sweep_duration_option == '7s':
        sweep_duration = 7
    elif sweep_duration_option == '14s':
        sweep_duration = 14

    if max_reverb_option == '1s':
        max_reverb_option = 1
    elif max_reverb_option == '2s':
        max_reverb_option = 2
    elif max_reverb_option == '3s':
        max_reverb_option = 3
    elif max_reverb_option == '5s':
        max_reverb_option = 5
    elif max_reverb_option == '10s':
        max_reverb_option = 10

    return sweep_duration_option, max_reverb_option


def upload_to_bucket(blob_name, path_to_file, bucket_name):
    """ Upload data to a bucket"""

    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        r'streamlit-project-files-727a2bb135cd.json')

    # print(buckets = list(storage_client.list_buckets())

    bucket = storage_client.get_bucket("streamlit-project-bucket")
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)

    # returns a public url
    return blob.public_url


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    bucket_name = "streamlit-project-bucket"
    # The path to your file to upload
    source_file_name = r'data/audio_files'
    # The ID of your GCS object
    destination_blob_name = "streamlit-project-bucket"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


def upload_from_directory(directory_path: str, dest_bucket_name: str, dest_blob_name: str):
    rel_paths = glob.glob(directory_path + '/**', recursive=True)
    bucket = GCS_CLIENT.get_bucket(dest_bucket_name)
    for local_file in rel_paths:
        remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def write_wav_file(file_name, rate, data):
    audio_files_path = r'data/audio_files'
    """Write wav file base on input"""
    save_file_path = os.path.join(audio_files_path, file_name)

    wavfile.write(save_file_path, rate, data)
    st.success(
        f"File successfully written to audio_files_path as:>> {file_name}")


def sweep_save():
    playbtn = st.button("Play")
    if "playbtn_state" not in st.session_state:
        st.session_state.playbtn_state = False

    if playbtn or st.session_state.playbtn_state:
        st.session_state.playbtn_state = True

        sweep = generate_exponential_sweep(sweep_duration, sample_rate)
        inverse = generate_inverse_filter(
            sweep_duration, sample_rate, sweep)
        ir = deconvolve(sweep, inverse)

        user_input = str(st.text_input("Name your file: "))

        if user_input:
            sweep_string = user_input + "_exponential_sweep_.wav"
            inv_filter_string = user_input + "_inverse_filter_.wav"
            ir_string = user_input + "_impulse_response_.wav"

            write_wav_file(file_name=sweep_string,
                           rate=sample_rate, data=sweep)

            write_wav_file(file_name=inv_filter_string,
                           rate=sample_rate, data=inverse)

            write_wav_file(file_name=ir_string, rate=sample_rate, data=ir)

        # upload_from_directory(audio_files_path, "streamlit-project-bucket", sweep_string)
        """
        upload_from_directory(audio_files_path, "streamlit-project-bucket", user_inverse_filter_string)
        upload_from_directory(audio_files_path, "streamlit-project-bucket", user_ir_string)
        upload_to_bucket(user_sweep_string, audio_files_path,
                         "streamlit-project-bucket")
        upload_to_bucket(user_inverse_filter_string,
                         audio_files_path, "streamlit-project-bucket")
        upload_to_bucket(user_ir_string, audio_files_path,
                         "streamlit-project-bucket")
        """


def play_sweep():
    CHUNK = 1024

    if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
        sys.exit(-1)

    wf = wave.open(sys.argv[1], 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while len(data):
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()


def irm_tab():
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["Impulse",
         "ETC",
         "Schroeder Integral",
         "EDT",
         "T20",
         "T30",
         "Clarity C50",
         "Clarity C80",
         "FFT"]
    )

    with tab1:
        st.header("Impulse")
        st.markdown(
            """
            The impulse plot shows the decay of the IR visually.
            """
        )

        # asyncio.run(time_convert())
        select_sweep_time()
        sweep_save()

    with tab2:
        st.header("ETC")
        st.markdown(
            """
            This is the time-domain plot of the decay response  of the IR
            """
        )

    with tab3:
        st.header("Schroeder Integral")
        st.markdown(
            """
            The backwards integration of the decay function of the impulse response.
            """
        )

    with tab4:
        st.header("EDT")
        st.markdown(
            """
            Early decay time represents the decay function of the IR in the slope of the early part of the energy decay curve.
            It is the slope of the curve limited from 0 dB to -10 dB extracted to a decay of 60 dB below
            to stopping of the direct sound energy.
            EDT is typically associated with the perceived reverberation time in a room.
            """
        )

    with tab5:
        st.header("T20")
        st.markdown(
            """
            T20 is the decay rate of the impulse response or reverberation
            time, the time that is takes for a sound to decay by 60 dB in a space.
            By using a noise compensation technique to avoid inaccurate representations of reverberation time,
            the decay rate is calculated by determining the slope of the decay function from -5 dB to -25 dB,
            and extrapolating the time to a decay of 60 dB.
            """
        )

    with tab6:
        st.header("T30")
        st.markdown(
            """
            T30 is the decay rate of the impulse response or reverberation time,
            the time that is takes for a sound to decay by 60 dB in a space.
            By using a noise compensation technique to avoid inaccurate representations of reverberation time,
            the decay rate is calculated by determining the slope of the decay function from -5 dB to -35 dB,
            and extrapolating the time to a decay of 60 dB.
            """
        )

    with tab7:
        st.header("Clarity C50")
        st.markdown(
            """
            The Clarity Factor (50ms), expressed in decibels, is the ratio of the early energy
            (0-50 ms) to the late reverberant energy (50-end of the decay of the IR).
            C50 is perceptually aligned with speech perception.
            """
        )

    with tab8:
        st.header("Clarity C80")
        st.markdown(
            """
            The Clarity Factor (80ms), expressed in decibels, is the ratio of the early energy
            (0-80 ms) to the late reverberant energy (80-end of the decay of the IR).
            C80 is perceptually aligned with music perception.
            """
        )

    with tab9:
        st.header("FFT")
        st.markdown(
            """
            The FFT of the ETC results in the frequency response graph for the signal.
            """
        )
