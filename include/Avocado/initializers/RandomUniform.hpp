/*
 * RandomUniform.hpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_INITIALIZERS_RANDOMUNIFORM_HPP_
#define AVOCADO_INITIALIZERS_RANDOMUNIFORM_HPP_

#include <Avocado/initializers/Initializer.hpp>

namespace avocado
{

	class RandomUniform: public Initializer
	{
			float m_min = -0.05f;
			float m_max = 0.05f;
		public:
			RandomUniform(float min_value = -0.05f, float max_value = 0.05f);

			void init(Parameter &param);

			std::string name() const;
			RandomUniform* clone() const;
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace avocado */

#endif /* AVOCADO_INITIALIZERS_RANDOMUNIFORM_HPP_ */
