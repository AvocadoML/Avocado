/*
 * RegularizerL2.cpp
 *
 *  Created on: Oct 13, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/regularizers/RegularizerL2.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/layers/Parameter.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/static_block.hpp>
#include <Avocado/math/training.hpp>

namespace avocado
{
	static_block
	{
//		registerRegularizer(RegularizerL2());
	}

	RegularizerL2::RegularizerL2(float coefficient, float offset) :
			m_coefficient(coefficient),
			m_offset(offset)
	{
	}

	void RegularizerL2::apply(const Context &context, Parameter &param)
	{

	}

	std::string RegularizerL2::name() const
	{
		return "RegularizerL2";
	}
	RegularizerL2* RegularizerL2::clone() const
	{
		return new RegularizerL2(this->m_coefficient, this->m_offset);
	}
	Json RegularizerL2::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = name();
		result["coefficient"] = m_coefficient;
		result["offset"] = m_offset;
		return result;
	}
	void RegularizerL2::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_coefficient = json["coefficient"];
		m_offset = json["offset"];
	}
} /* namespace avocado */

