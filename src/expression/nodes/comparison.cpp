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

		Equal* Equal::clone() const
		{
			return new Equal();
		}
		std::string Equal::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " == " + getInput(1).text();
		}

		NotEqual* NotEqual::clone() const
		{
			return new NotEqual();
		}
		std::string NotEqual::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " != " + getInput(1).text();
		}

		LowerThan* LowerThan::clone() const
		{
			return new LowerThan();
		}
		std::string LowerThan::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " < " + getInput(1).text();
		}

		LowerOrEqual* LowerOrEqual::clone() const
		{
			return new LowerOrEqual();
		}
		std::string LowerOrEqual::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " <= " + getInput(1).text();
		}

		GreaterThan* GreaterThan::clone() const
		{
			return new GreaterThan();
		}
		std::string GreaterThan::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " > " + getInput(1).text();
		}

		GreaterOrEqual* GreaterOrEqual::clone() const
		{
			return new GreaterOrEqual();
		}
		std::string GreaterOrEqual::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " >= " + getInput(1).text();
		}

	} /* namespace nodes */
} /* namespace avocado */

