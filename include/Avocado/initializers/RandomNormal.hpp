/*
 * RandomNormal.hpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_INITIALIZERS_RANDOMNORMAL_HPP_
#define AVOCADO_INITIALIZERS_RANDOMNORMAL_HPP_

#include <Avocado/initializers/Initializer.hpp>

namespace avocado
{

	class RandomNormal: public Initializer
	{
			float m_mean = 0.0f;
			float m_stddev = 1.0f;
		public:
			RandomNormal(float mean = 0.0f, float stddev = 1.0f);

			void init(Parameter &param);

			std::string name() const;
			RandomNormal* clone() const;
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace avocado */

#endif /* AVOCADO_INITIALIZERS_RANDOMNORMAL_HPP_ */
