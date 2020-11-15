import numpy as np
from mne.filter import filter_data, next_fast_len
from scipy.signal import hilbert
from multiprocessing import Pool
import timeit
from ConnFunc import *

class Signal():
	"""This class handles the signal analysis - band pass filtering,
	amplitude extraction and low-pass filtering the amplitude 
	"""
	def __init__(self, mat, fsample=None, lowpass=None):
		"""
		Initialize the signal object with a signal matrix
		:param mat: n (regions) x p (timepoints) numpy ndarray containing the signal
		"""
		assert isinstance(mat, np.ndarray), "Signal must be numpy array"
		self.Signal = mat
		self.fsample = fsample
		self.lowpass = lowpass
		self.NumberRegions, self.TimePoints = mat.shape		

	def __getitem__(self, index):
		return self.Signal[index]

	def getFrequencyBand(self, Limits):
		"""
		Band pass filters signal from each region
		:param Limits: int, specifies limits of the frequency band
		:return: filter Signal
		"""
		lowerfreq = Limits[0]
		upperfreq = Limits[1]
		filteredSignal = filter_data(self.Signal, self.fsample, l_freq=lowerfreq, h_freq=upperfreq,
									 fir_window='hamming', verbose=False)
		return filteredSignal

	def getEnvelope(self, Limits):
		filteredSignal = self.getFrequencyBand(Limits)
		n_fft = next_fast_len(self.TimePoints)
		complex_signal = hilbert(filteredSignal, N=n_fft, axis=-1)
		filteredEnvelope = np.abs(complex_signal)
		return filteredEnvelope

	def resampleSignal(self, resample_num=None, TargetFreq=None):
		"""
		Resamples Signal to number of resample points or to target frequency
		"""
		from scipy.signal import resample
		if TargetFreq is not None:
			downsamplingFactor = TargetFreq/self.fsample
			resample_num = int(self.Signal.shape[1]*downsamplingFactor)

		if self.Signal.shape[1] < resample_num:
			raise Exception('Target sample size should be smaller than original frequency')
		
		# Resample
		re_signal = resample(self.Signal, num=resample_num, axis=-1)
		
		# Save to signal
		self.Signal = re_signal

		# Reevaluate Sampling Frequency
		downsamplingFactor = resample_num / self.TimePoints
		self.fsample = self.fsample * downsamplingFactor
		self.NumberRegions, self.TimePoints = self.Signal.shape
		return self.Signal

	def getFC(self, Limits, conn_mode):
		"""
		Computes the Functional Connectivity Matrix based on the signal envelope
		Takes settings from configuration File. If conn_mode contains 
		orth the signal is orthogonalized in parallel using _parallel_orth_corr
		"""
		# Filter signal
		FilteredSignal = self.getFrequencyBand(Limits)

		# Get complex signal
		n_fft = next_fast_len(self.TimePoints)
		ComplexSignal = hilbert(FilteredSignal, N=n_fft, axis=-1)[:, :self.TimePoints]
		pad=100
		ComplexSignal = ComplexSignal[:,pad:-pad]

		# Get signal envelope
		SignalEnv = np.abs(ComplexSignal)
		
		# If no conn_mode is specified, unorthogonalized FC is computed.
		if conn_mode in ['lowpass-corr', 'corr']:			
			if 'lowpass' in conn_mode: 
				SignalEnv = filter_data(SignalEnv, self.fsample, 0, self.lowpass, fir_window='hamming', verbose=False)			
			FC = pearson(SignalEnv, SignalEnv)
			return FC

		SignalConj = ComplexSignal.conj()
		ConjdivEnv = SignalConj/SignalEnv 

		# Compute orthogonalization and correlation in parallel		
		with Pool(processes=10) as p: 
			result = p.starmap(self._parallel_orth_corr, [(Complex, SignalEnv, ConjdivEnv) for Complex in ComplexSignal])
		FC = np.array(result)

		# Make the Corr Matrix symmetric
		FC = (FC.T + FC) / 2.
		return FC
	
	def _parallel_orth_corr(self, ComplexSignal, SignalEnv, ConjdivEnv):
		"""
		Computes orthogonalized correlation of the envelope of the complex signal (nx1 dim array) and the signal envelope  (nxm dim array). 
		This function is called by signal.getOrthFC()
		:param ComplexSignal Complex 
		"""
		# Orthogonalize signal
		OrthSignal = (ComplexSignal * ConjdivEnv).imag
		OrthEnv = np.abs(OrthSignal)
		# Envelope Correlation
		if 'lowpass' in conn_mode:
			# Low-Pass filter
			OrthEnv = filter_data(OrthEnv, self.fsample, 0, self.lowpass, fir_window='hamming', verbose=False)
			SignalEnv = filter_data(SignalEnv, self.fsample, 0, self.lowpass, fir_window='hamming', verbose=False)	
		corr_mat = pearson(OrthEnv, SignalEnv)	
		corr = np.diag(corr_mat)
		return corr

	def getOrthEnvelope(self, Index, ReferenceIndex, FreqBand, pad=100, LowPass=False):
		"""
		Function to compute the Orthogonalized Envelope of the indexed signal with respect to a reference signal.
		Is used create a plot the orthogonalized Envelope.
		"""
		Limits = FreqBand
		# Filter signal
		FilteredSignal = self.getFrequencyBand(Limits)
		# Get complex signal
		n_fft = next_fast_len(self.TimePoints)
		ComplexSignal = hilbert(FilteredSignal, N=n_fft, axis=-1)[:, :self.TimePoints]
		ComplexSignal = ComplexSignal[:,pad:-pad]

		# Get signal envelope and conjugate
		SignalEnv = np.abs(ComplexSignal)
		SignalConj = ComplexSignal.conj()

		OrthSignal = (ComplexSignal[Index] * (SignalConj[ReferenceIndex] / SignalEnv[ReferenceIndex])).imag
		OrthEnv = np.abs(OrthSignal)

		if LowPass:
			OrthEnv = filter_data(OrthEnv, self.fsample, 0, self.lowpass, fir_window='hamming', verbose=False)
			ReferenceEnv = filter_data(SignalEnv[ReferenceIndex], self.fsample, 0, self.lowpass, fir_window='hamming', verbose=False)
		else:
			ReferenceEnv = SignalEnv

		return OrthEnv, ReferenceEnv

	def getMetastability(self, Limits):
		"""
		Computes Kuramoto Parameter of low pass envelope and Metastability as standard deviation 
		of the Kuramoto Parameter. 
		:params Frequency Band Limits to compute the Envelope
		"""
		envelope = self.getEnvelope(Limits=Limits)
		# Demean envelope
		envelope -= np.mean(envelope, axis=-1, keepdims=True)
		# Low pass filter envelope
		l_envelope = Signal(envelope, fsample=self.fsample).getFrequencyBand(Limits=[0, self.lowpass])
		
		# Compute the Kuramoto Parameter
		analytic = hilbert(l_envelope, axis=-1)
		phase = np.angle(analytic)
		ImPhase = phase * 1j
		SumPhase = np.sum(np.exp(ImPhase), axis=0) 
		Kuramoto = np.abs(SumPhase) / ImPhase.shape[0]

		# Metastability is standard deviation of Kuramoto Parameter  
		Metastability = np.std(Kuramoto, axis=-1)
		
		return Kuramoto, Metastability

	def getCCD(self, Limits, DownFreq=5):
		""" Defines the Coherence Connectivity Dynamics following the description of Deco et. al 2017
		:param signal: numpy ndarray containing the signal envelope
		:return: numpy nd array containing the CCD matrix
		"""
		envelope = self.getEnvelope(Limits=Limits)
		# Demean envelope
		envelope -= np.mean(envelope, axis=-1, keepdims=True)
		# Low pass filter envelope
		l_envelope = Signal(envelope, fsample=self.fsample).getFrequencyBand(Limits=[0, self.lowpass])
		# Call the downsampling function twice to downsample to DownFreq
		ld_envelope = Signal(l_envelope, fsample=self.fsample).resampleSignal(TargetFreq=DownFreq)
		# Calculate V and CCD. 
		V = self._calc_v(ld_envelope)
		#V = self._smooth_mat(V,width=10)
		CCD = self._calc_ccd(V)
		return CCD.astype('float32')
	
	def _calc_v(self, signal): 
		"""
		Computes V matrix which is used to compute the CCD matrix. 
		V contains the cosine phase difference between all pairs of the 
		signal.
		:return V mat
		"""
		#calculate phase of envelope
		analytic = hilbert(signal, axis=-1)
		phases = np.angle(analytic)

		#Calculate phase differences 
		diff_phases = [np.abs(phase1-phase2) for phase1 in phases for phase2 in phases]
		diff_phases = np.stack(diff_phases)

		# V is the cosine of the absolute phase differences
		v = np.cos(diff_phases)
		return v
	
	def _calc_ccd(self, v):
		"""
		Computes CCD matrix of signal. This function is called by getCCD. 
		Uses vector norm to calculate the cosine similarity 
		:params V mat which is computed in _calc_v
		"""
		from scipy.linalg import norm
		# normalize matrix along first axis
		v_norm = norm(v, axis=0)
		# Calculate the ccd as normalized vector product of all 
		# column vectors in v matrix
		ccd = np.matmul(v.T, v)
		ccd /= v_norm 
		ccd = ccd.T / v_norm 
		return ccd 
	
	def _smooth_mat(self, mat, width):
		"""
		Takes mean of valuew in windows with fixed width.
		"""
		mat = mat[:,:width*int(mat.shape[1]//width)]
		re_mat = mat.reshape(-1, int(mat.shape[1]/width), width)
		mean_mat = np.mean(re_mat, axis=-1)
		return mean_mat
		
