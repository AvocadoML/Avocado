/*
 * activations.cpp
 *
 *  Created on: Aug 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/nodes/activations.hpp>
#include <Avocado/expression/Expression.hpp>

namespace avocado
{
	namespace nodes
	{

		std::string Sigmoid::toString() const
		{
			return this->text() + " = sigmoid(" + getInput(0).text() + ")";
		}
		Expression Sigmoid::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto y = result.view(this);
			result.output(dy * y * (result.one() - y));
			return result;
		}

		std::string ReLU::toString() const
		{
			return this->text() + " = relu(" + getInput(0).text() + ")";
		}
		Expression ReLU::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto y = result.view(this);
			result.output(result.select(y > result.zero(), dy, result.zero()));
			return result;
		}

	} /* namespace nodes */
} /* namespace avocado */

