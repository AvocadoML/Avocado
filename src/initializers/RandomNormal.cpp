/*
 * RandomNormal.cpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/initializers/RandomNormal.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/math/random.hpp>
#include <Avocado/layers/Parameter.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
		registerInitializer(RandomNormal());
	}

	RandomNormal::RandomNormal(float mean, float stddev) :
			m_mean(mean),
			m_stddev(stddev)
	{
	}

	void RandomNormal::init(Parameter &param)
	{
		float scale;
		if (param.shape().length() == 1)
			scale = m_stddev / sqrt(param.shape().firstDim());
		else
			scale = m_stddev / sqrt(param.shape().volumeWithoutFirstDim());

		const size_t volume = param.shape().volume();
		std::unique_ptr<float[]> tmp = std::make_unique<float[]>(volume);
		for (size_t i = 0; i < volume; i++)
			tmp[i] = scale * math::randGaussian() + m_mean;
		param.getParam().copyFromHost(tmp.get(), param.shape().volume());
	}

	std::string RandomNormal::name() const
	{
		return "RandomNormal";
	}
	RandomNormal* RandomNormal::clone() const
	{
		return new RandomNormal(this->m_mean, this->m_stddev);
	}
	Json RandomNormal::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = name();
		result["mean"] = m_mean;
		result["stddev"] = m_stddev;
		return result;
	}
	void RandomNormal::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_mean = json["mean"];
		m_stddev = json["stddev"];
	}

} /* namespace avocado */

