/*
 * bitwise.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/nodes/bitwise.hpp>
#include <Avocado/expression/Expression.hpp>

namespace avocado
{
	namespace nodes
	{

		std::string BitwiseNot::toString() const
		{
			return this->text() + " = ~" + getInput(0).text();
		}

		std::string BitwiseAnd::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " & " + getInput(1).text();
		}

		std::string BitwiseOr::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " | " + getInput(1).text();
		}

		std::string BitwiseXor::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " ^ " + getInput(1).text();
		}

		std::string BitwiseShiftLeft::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " << " + getInput(1).text();
		}

		std::string BitwiseShiftRight::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " >> " + getInput(1).text();
		}

	} /* namespace nodes */
} /* namespace avocado */

