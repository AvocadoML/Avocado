/*
 * calibration.cpp
 *
 *  Created on: Jun 17, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/inference/calibration.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/serialization.hpp>
#include <Avocado/math/random.hpp>

#include <numeric>
#include <algorithm>
#include <omp.h>

namespace
{
	double histogram_diff(const std::vector<double> &lhs, const std::vector<double> &rhs)
	{
		assert(lhs.size() == rhs.size());

		double area_lhs = std::accumulate(lhs.begin(), lhs.end(), 0.0) + 1e-16;
		double area_rhs = std::accumulate(rhs.begin(), rhs.end(), 0.0) + 1e-16;

		double diff = 0.0;
		for (size_t i = 0; i < lhs.size(); i++)
			diff += fabs(lhs[i] / area_lhs - rhs[i] / area_rhs);
		return diff;
	}
	void resize_histogram(std::vector<double> &histogram, std::vector<double> &workspace, size_t from_bin, size_t to_bin)
	{
		for (size_t i = 0; i < from_bin; i++)
		{
			histogram[from_bin] += histogram[i];
			histogram[i] = 0;
		}
		for (size_t i = to_bin; i < histogram.size(); i++)
		{
			histogram[to_bin - 1] += histogram[i];
			histogram[i] = 0;
		}

		workspace.assign(workspace.size(), 0);
		const size_t orig_size = to_bin - from_bin;
		for (size_t i = 0; i < workspace.size(); i++)
		{
			size_t start_idx = from_bin + i * orig_size / workspace.size();
			size_t end_idx = from_bin + ((i + 1) * orig_size + workspace.size() - 1) / workspace.size();

			double idx0 = from_bin + static_cast<double>(i) * orig_size / workspace.size();
			double idx1 = from_bin + static_cast<double>(i + 1) * orig_size / workspace.size();

			for (size_t j = start_idx; j < end_idx; j++)
			{
				double weight = std::min(static_cast<double>(j + 1), idx1) - std::max(static_cast<double>(j), idx0);
				workspace[i] += histogram[j] * weight;
			}
		}

		histogram.assign(histogram.size(), 0);
		const double bin_width = static_cast<double>(to_bin - from_bin) / workspace.size();
		for (size_t i = 0; i < workspace.size(); i++)
		{
			size_t start_idx = from_bin + i * orig_size / workspace.size();
			size_t end_idx = from_bin + ((i + 1) * orig_size + workspace.size() - 1) / workspace.size();

			double idx0 = from_bin + static_cast<double>(i) * orig_size / workspace.size();
			double idx1 = from_bin + static_cast<double>(i + 1) * orig_size / workspace.size();

			for (size_t j = start_idx; j < end_idx; j++)
			{
				double weight = std::min(static_cast<double>(j + 1), idx1) - std::max(static_cast<double>(j), idx0);
				histogram[j] += workspace[i] * weight / bin_width;
			}
		}
	}
	double cross_entropy(const std::vector<double> &lhs, const std::vector<double> &rhs)
	{
		assert(lhs.size() == rhs.size());
		double result = 0.0;
		for (size_t i = 0; i < lhs.size(); i++)
			result += lhs[i] * std::log(rhs[i] + 1e-16) + (1.0 - lhs[i]) * std::log(1.0 - rhs[i] + 1e-16);
		return -result;
	}
	template<typename T>
	void print_dist(const std::vector<T> &dist)
	{
		for (size_t i = 0; i < dist.size(); i++)
			std::cout << dist[i] << ' ';
		std::cout << '\n';
	}
	std::vector<double> normalize(const std::vector<double> &v)
	{
		const double scale = 1.0 / std::accumulate(v.begin(), v.end(), 0.0) + 1e-16;
		std::vector<double> result(v.size(), 0.0);
		for (size_t i = 0; i < result.size(); i++)
			result[i] = v[i] * scale;
		return result;
	}
}

namespace avocado
{
	namespace inference
	{
		Histogram::Histogram(int numberOfBins, float accuracy) :
				m_data(numberOfBins, 0),
				m_accuracy(accuracy)
		{
		}
		void Histogram::collectStatistics(const Tensor &tensor)
		{
			Tensor tmp(tensor.shape(), tensor.dtype(), Device::cpu());
			tmp.copyFrom(tensor);

			if (not has_enough_samples_for_min_max())
				find_min_max(tensor);

			// intentionally not in else block, so after min/max condition is met, we reuse the same tensor for histogram collection
			if (has_enough_samples_for_min_max())
			{
				std::vector<double> previous_histogram = m_data;
				create_histogram(tmp);
				double diff = histogram_diff(previous_histogram, m_data) / pow(tensor.firstDim(), 0.666f);
				if (diff < m_accuracy)
					m_is_ready = true;
			}
		}
		bool Histogram::isReady() const noexcept
		{
			return m_is_ready;
		}
		std::string Histogram::getInfo() const
		{
			std::string result;
			result += "min = " + std::to_string(m_min_value) + ", max = " + std::to_string(m_max_value) + '\n';
//		for (size_t i = 0; i < m_data.size(); i++)
//			result += "from " + std::to_string(m_min_value + (m_max_value - m_min_value) * i / m_data.size()) + " to "
//					+ std::to_string(m_min_value + (m_max_value - m_min_value) * (i + 1) / m_data.size()) + " = " + std::to_string(m_data[i]) + '\n';
			return result;
		}
		Json Histogram::serialize(SerializedObject &binary_data) const
		{
			Json result;
			result["min"] = m_min_value;
			result["max"] = m_max_value;
			result["binary_offset"] = binary_data.size();
			binary_data.save(m_data.data(), sizeof(size_t) * m_data.size());
			return result;
		}
		void Histogram::quantizeTo(size_t bins) const
		{
			std::vector<double> workspace(bins);
			std::vector<double> normalized = normalize(m_data);
			std::vector<double> copy;

			const double entropy = cross_entropy(normalized, normalized);
			int best_start = 0;
			int best_end = m_data.size();
			double best_value = std::numeric_limits<double>::max();
			for (size_t i = 0; i <= m_data.size() - bins; i += 32)
				for (size_t j = i + bins; j <= m_data.size(); j += 32)
				{
					copy = normalized;
					resize_histogram(copy, workspace, i, j);
					double tmp = cross_entropy(normalized, copy) - entropy;
					if (tmp < best_value)
					{
						best_start = i;
						best_end = j;
						best_value = tmp;
					}
				}
		}
		void Histogram::find_min_max(const Tensor &tensor)
		{
			assert(tensor.device().isCPU());

			std::unique_ptr<float[]> tensor_data = toArray<float>(tensor);
			m_outliers_count = 0;
			float min_value = m_min_value;
			float max_value = m_max_value;
			for (int i = 0; i < tensor.volume(); i++)
			{
				min_value = std::min(min_value, tensor_data[i]);
				max_value = std::max(max_value, tensor_data[i]);
				if (tensor_data[i] < m_min_value or tensor_data[i] > m_max_value)
					m_outliers_count++;
			}
			m_collected_samples += tensor.volume();

			m_min_value = min_value;
			m_max_value = max_value;
		}
		void Histogram::create_histogram(const Tensor &tensor)
		{
			assert(tensor.device().isCPU());

			std::unique_ptr<float[]> tensor_data = toArray<float>(tensor);
			for (int i = 0; i < tensor.volume(); i++)
			{
				size_t bin_index = (tensor_data[i] - m_min_value) / (m_max_value - m_min_value + 1e-16f) * m_data.size();
				m_data[std::max(0.0, std::min(m_data.size() - 1.0, static_cast<double>(bin_index)))]++;
			}
		}
		bool Histogram::has_enough_samples_for_min_max() const noexcept
		{
			return static_cast<float>(m_outliers_count) / m_collected_samples < m_accuracy;
		}

		CalibrationTable::CalibrationTable(int numberOfBins, float accuracy) noexcept :
				m_number_of_bins(numberOfBins),
				m_accuracy(accuracy)
		{
		}
		Histogram& CalibrationTable::getHistogram(size_t index)
		{
			if (index >= m_histograms.size())
				m_histograms.insert(m_histograms.end(), 1 + index - m_histograms.size(), Histogram(m_number_of_bins, m_accuracy));
			return m_histograms.at(index);
		}
		const Histogram& CalibrationTable::getHistogram(size_t index) const
		{
			return m_histograms.at(index);
		}
		size_t CalibrationTable::size() const noexcept
		{
			return m_histograms.size();
		}
		bool CalibrationTable::isReady() const noexcept
		{
			if (m_histograms.empty())
				return false;
			else
				return std::all_of(m_histograms.begin(), m_histograms.end(), [](const auto &h)
				{	return h.isReady();});
		}
		Json CalibrationTable::save(SerializedObject &binary_data) const
		{
			Json result(JsonType::Array);
			for (auto iter = m_histograms.begin(); iter < m_histograms.end(); iter++)
				result.append(iter->serialize(binary_data));
			return result;
		}
		void CalibrationTable::quantizeTo(size_t bins) const
		{
#pragma omp parallel for
			for (auto iter = m_histograms.begin(); iter < m_histograms.end(); iter++)
				iter->quantizeTo(bins);
		}
	} /* namespace inference */
} /* namespace avocado */

