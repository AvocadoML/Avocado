/*
 * arithmetic.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/nodes/arithmetic.hpp>
#include <Avocado/expression/Expression.hpp>
#include <Avocado/core/error_handling.hpp>

#include <cassert>

namespace avocado
{
	namespace nodes
	{

		std::string Negation::toString() const
		{
			return this->text() + " = -" + getInput(0).text();
		}
		Expression Negation::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			result.output(-dy);
			return result;
		}

		std::string Addition::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " + " + getInput(1).text();
		}
		Expression Addition::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			result.output(dy);
			result.output(dy);
			return result;
		}

		std::string Subtraction::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " - " + getInput(1).text();
		}
		Expression Subtraction::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			result.output(dy);
			result.output(-dy);
			return result;
		}

		std::string Multiplication::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " * " + getInput(1).text();
		}
		Expression Multiplication::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x1 = result.view(m_inputs.at(0));
			auto x2 = result.view(m_inputs.at(1));
			result.output(dy * x2);
			result.output(dy * x1);
			return result;
		}

		std::string Division::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " / " + getInput(1).text();
		}
		Expression Division::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x1 = result.view(m_inputs.at(0));
			auto x2 = result.view(m_inputs.at(1));
			result.output(dy / x2);
			result.output(-dy * x1 / result.square(x2));
			return result;
		}

		std::string Modulo::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " % " + getInput(1).text();
		}
		Expression Modulo::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x1 = result.view(m_inputs.at(0));
			auto x2 = result.view(m_inputs.at(1));
			result.output(dy);
			result.output(-dy * result.floor(x1 / x2));
			return result;
		}

	} /* namespace nodes */
} /* namespace avocado */

