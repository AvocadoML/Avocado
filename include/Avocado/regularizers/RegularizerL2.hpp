/*
 * RegularizerL2.hpp
 *
 *  Created on: Oct 13, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_REGULARIZERS_REGULARIZERL2_HPP_
#define AVOCADO_REGULARIZERS_REGULARIZERL2_HPP_

#include <Avocado/regularizers/Regularizer.hpp>

namespace avocado
{

	class RegularizerL2: public Regularizer
	{
			double m_scale = 0.0;
			double m_offset = 0.0;
		public:
			RegularizerL2() = default;
			RegularizerL2(double coefficient, double offset = 0.0f);

			void apply(const Context &context, Parameter &param);

			std::string name() const;
			RegularizerL2* clone() const;
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace avocado */

#endif /* AVOCADO_REGULARIZERS_REGULARIZERL2_HPP_ */
