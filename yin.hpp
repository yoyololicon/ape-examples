#include <effect.h>
#include <consts.h>
#include <numeric>

using namespace ape;

GlobalData(Yin, "");

class Yin : public Effect
{
public:
	Param<float> thresh{"Threshold", "dB", Range(-60, 6)};
	Param<float> tol{"Tolerance", Range(0, 0.3)};
	Param<float> out_gain{"Output Gain", "dB", Range(-60, 20)};
	Param<float> mix{"Mix", Range(0, 1)};
	Param<float> max_freq{"Max Frequency", "Hz", Range(800, 4000, Range::Exp)};
	Param<float> min_freq{"Min Frequency", "Hz", Range(40, 400, Range::Exp)};

	Param<bool> env{"Apply Input Envelope"};

	Yin()
	{
		tol = 0.1;
		thresh = -50;
		out_gain = -6;
		mix = 1;
		env = false;
		min_freq = 40;
	}

private:
	FFT<fpoint> fft;

	std::vector<fpoint> history;
	std::vector<fpoint> square_buf;
	std::vector<fpoint> diff;
	std::vector<fpoint> sin_table;

	std::vector<std::complex<fpoint>> fft_buf1;
	std::vector<std::complex<fpoint>> fft_buf2;

	size_t W, N, history_counter, i;

	fpoint sr, pitch, cur_phase, prev_incr, prev_signal, prev_norm;

	enum status
	{
		VOICED,
		UNVOICED,
		ENDING
	} state = UNVOICED;

	fpoint parabolic(fpoint fa, fpoint fb, fpoint fc)
	{
		return 0.5 * (fa - fc) / (fa + fc - 2 * fb);
	}

	void start(const IOConfig &cfg) override
	{
		sr = cfg.sampleRate;
		W = static_cast<int>(sr * 0.025);
		pitch = 0;
		cur_phase = 0;

		history.resize(2 * W - 1);
		square_buf.resize(2 * W - 1);
		diff.resize(W);

		sin_table.resize(2 * W);
		for (i = 0; i < 2 * W; i++)
			sin_table[i] = std::sin(i * consts<fpoint>::pi / W);

		N = nextpow2(2 * W - 1);

		fft = {N};

		fft_buf1.resize(N);
		fft_buf2.resize(N);

		print("W = %, sr = %", W, cfg.sampleRate);

		state = UNVOICED;
		prev_norm = 0;
	}

	void process(umatrix<const float> inputs, umatrix<float> outputs, size_t frames) override
	{

		std::copy(inputs[0].begin(), inputs[0].begin() + frames, cyclic_begin(history, history_counter));
		history_counter += frames;
		history_counter %= history.size();

		std::transform(cyclic_begin(history, history_counter),
					   cyclic_end(history, history_counter, history.size()),
					   square_buf.begin(), [](auto n) { return n * n; });

		fpoint tmp = std::accumulate(square_buf.begin(), square_buf.begin() + W, 0.);

		diff[0] = tmp;
		auto norm = tmp;

		for (i = W; i < 2 * W - 1; i++)
		{
			norm += square_buf[i];
			tmp += square_buf[i];
			tmp -= square_buf[i - W];
			diff[i - W + 1] = tmp;
		}

		norm = std::sqrt(norm / square_buf.size());
		auto norm_db = dB::to(norm);

		tmp = diff[0];

		for (auto &n : diff)
			n += tmp;

		std::copy(cyclic_begin(history, history_counter),
				  cyclic_end(history, history_counter, history.size()),
				  fft_buf1.begin());
		std::fill(fft_buf1.begin() + 2 * W - 1, fft_buf1.end(), 0);

		std::copy(fft_buf1.begin(), fft_buf1.begin() + W, fft_buf2.begin());
		std::fill(fft_buf2.begin() + W, fft_buf2.end(), 0);

		fft.forward(fft_buf1);
		fft.forward(fft_buf2);

		for (i = 0; i < N / 2 + 1; i++)
			fft_buf1[i] *= std::conj(fft_buf2[i]);

		for (; i < N; i++)
			fft_buf1[i] = std::conj(fft_buf1[N - i]);

		fft.inverse(fft_buf1);

		for (i = 0; i < W; i++)
			diff[i] -= 2 * std::real(fft_buf1[i]);

		tmp = 0;
		square_buf[0] = 1;
		for (i = 1; i < W; i++)
		{
			tmp += diff[i];
			square_buf[i] = diff[i] * i / tmp;
			//diff[i] *= i;
			//diff[i] /= tmp;
		}

		int tau = 0;
		fpoint p;
		const float maxFreqVal = max_freq;
		const float minFreqVal = min_freq;
		int smallest_lag = std::max(3, int(sr / maxFreqVal));
		int biggest_lag = int(sr / minFreqVal);

		for (i = smallest_lag + 1; i < biggest_lag + 1; i++)
		{
			if ((square_buf[i - 1] < tol) && (square_buf[i - 1] < square_buf[i]))
			{
				tau = i - 1;
				p = parabolic(diff[i - 2], diff[i - 1], diff[i]);
				break;
			}
		}

		fpoint incr;
		if (tau && (norm_db > thresh))
		{
			incr = sin_table.size();
			incr /= (p + tau);
			pitch = sr / (p + tau);
			if (state == UNVOICED)
			{
				cur_phase = 0;
				prev_incr = incr;
			}
			state = VOICED;
			print("pitch = %", pitch);
		}
		else
		{
			pitch = 0;
			incr = prev_incr;
			if (state == VOICED)
				state = ENDING;
		}

		fpoint signal;

		const auto shared = sharedChannels();

		const fpoint gain = out_gain;
		auto scale = dB::from(gain);

		const fpoint mixVal = mix;
		auto angle = (mixVal - 0.5) * consts<fpoint>::pi_half;
		auto cos = std::cos(angle);
		auto sin = std::sin(angle);
		auto root2over2 = std::sqrt(2.0) * 0.5;
		auto input_scale = root2over2 * (cos - sin);
		auto output_scale = root2over2 * (cos + sin);

		for (std::size_t n = 0; n < frames; ++n)
		{
			p = static_cast<fpoint>(n) / frames;
			cur_phase += incr * p + (1 - p) * prev_incr;
			signal = sin_table[static_cast<int>(cur_phase + 0.5) % sin_table.size()];
			if (env)
			{
				signal *= (prev_norm * (1 - p) + p * norm) / 0.707;
			}
			signal *= scale;

			if (state == ENDING)
			{
				if (prev_signal * signal < 0)
					state = UNVOICED;
			}
			if (state == UNVOICED)
				signal = 0;
			for (std::size_t c = 0; c < shared; ++c)
				outputs[c][n] = signal * output_scale + inputs[c][n] * input_scale;

			prev_signal = signal;
		}

		while (cur_phase >= sin_table.size())
			cur_phase -= sin_table.size();

		prev_norm = norm;
		if (state == VOICED)
			prev_incr = incr;
		clear(outputs, shared);
	}
};
