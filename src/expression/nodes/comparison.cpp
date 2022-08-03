/*
 * comparison.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/nodes/comparison.hpp>
#include <Avocado/expression/Expression.hpp>

namespace avocado
{
	namespace nodes
	{

		std::string Equal::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " == " + getInput(1).text();
		}

		std::string NotEqual::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " != " + getInput(1).text();
		}

		std::string LowerThan::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " < " + getInput(1).text();
		}

		std::string LowerOrEqual::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " <= " + getInput(1).text();
		}

		std::string GreaterThan::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " > " + getInput(1).text();
		}

		std::string GreaterOrEqual::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " >= " + getInput(1).text();
		}

	} /* namespace nodes */
} /* namespace avocado */

