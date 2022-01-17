/*
 * calibration.hpp
 *
 *  Created on: Jun 12, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_INFERENCE_CALIBRATION_HPP_
#define AVOCADO_INFERENCE_CALIBRATION_HPP_

#include <vector>
#include <inttypes.h>
#include <limits>
#include <string>

namespace avocado /* forward declarations */
{
	class Tensor;
	class Json;
	class SerializedObject;
}

namespace avocado
{
	namespace inference
	{
		class Histogram
		{
			private:
				std::vector<double> m_data;
				float m_min_value = std::numeric_limits<float>::max();
				float m_max_value = std::numeric_limits<float>::lowest();
				size_t m_collected_samples = 0;
				size_t m_outliers_count = 0;

				float m_accuracy;
				bool m_is_ready = false;
			public:
				explicit Histogram(int numberOfBins, float accuracy);
				void collectStatistics(const Tensor &tensor);
				bool isReady() const noexcept;
				std::string getInfo() const;
				Json serialize(SerializedObject &binary_data) const;
				void quantizeTo(size_t bins) const;
			private:
				void find_min_max(const Tensor &tensor);
				void create_histogram(const Tensor &tensor);
				bool has_enough_samples_for_min_max() const noexcept;
		};

		class CalibrationTable
		{
			private:
				std::vector<Histogram> m_histograms;
				int m_number_of_bins;
				float m_accuracy;
			public:
				explicit CalibrationTable(int numberOfBins = 2048, float accuracy = 1.0e-3f) noexcept;

				Histogram& getHistogram(size_t index);
				const Histogram& getHistogram(size_t index) const;
				size_t size() const noexcept;

				bool isReady() const noexcept;
				Json save(SerializedObject &binary_data) const;
				void quantizeTo(size_t bins) const;
		};
	} /* namespace inference */
} /* namespace avocado */

#endif /* AVOCADO_INFERENCE_CALIBRATION_HPP_ */
