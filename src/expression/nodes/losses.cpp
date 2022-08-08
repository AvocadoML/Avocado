/*
 * losses.cpp
 *
 *  Created on: Jul 31, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/nodes/losses.hpp>
#include <Avocado/expression/Expression.hpp>

namespace avocado
{
	namespace nodes
	{

		MeanSquareLoss* MeanSquareLoss::clone() const
		{
			return new MeanSquareLoss();
		}
		std::string MeanSquareLoss::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " * " + getInput(1).text();
		}
		Expression MeanSquareLoss::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x1 = result.view(m_inputs.at(0));
			auto x2 = result.view(m_inputs.at(1));
			result.output(dy * (x1 - x2));
			result.output(dy * (x2 - x1));
			return result;
		}

	} /* namespace nodes */
} /* namespace avocado */
