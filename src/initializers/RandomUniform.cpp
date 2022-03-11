/*
 * RandomUniform.cpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/initializers/RandomUniform.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/layers/Parameter.hpp>
#include <Avocado/math/random.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
		registerInitializer(RandomUniform());
	}

	RandomUniform::RandomUniform(float min_value, float max_value) :
			m_min(min_value),
			m_max(max_value)
	{
	}

	void RandomUniform::init(Parameter &param)
	{
		if (param.shape().volume() == 0)
			return;
		const size_t volume = param.shape().volume();
		std::unique_ptr<float[]> tmp = std::make_unique<float[]>(volume);
		for (size_t i = 0; i < volume; i++)
			tmp[i] = m_min + (m_max - m_min) * math::randFloat();
		param.getParam().copyFromHost(tmp.get(), param.shape().volume());
	}

	std::string RandomUniform::name() const
	{
		return "RandomUniform";
	}
	RandomUniform* RandomUniform::clone() const
	{
		return new RandomUniform(this->m_min, this->m_max);
	}
	Json RandomUniform::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = name();
		result["min"] = m_min;
		result["max"] = m_max;
		return result;
	}
	void RandomUniform::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_min = json["min"];
		m_max = json["max"];
	}

} /* namespace avocado */

