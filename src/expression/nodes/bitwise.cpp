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

		BitwiseNot* BitwiseNot::clone() const
		{
			return new BitwiseNot();
		}
		std::string BitwiseNot::toString() const
		{
			return this->text() + " = ~" + getInput(0).text();
		}

		BitwiseAnd* BitwiseAnd::clone() const
		{
			return new BitwiseAnd();
		}
		std::string BitwiseAnd::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " & " + getInput(1).text();
		}

		BitwiseOr* BitwiseOr::clone() const
		{
			return new BitwiseOr();
		}
		std::string BitwiseOr::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " | " + getInput(1).text();
		}

		BitwiseXor* BitwiseXor::clone() const
		{
			return new BitwiseXor();
		}
		std::string BitwiseXor::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " ^ " + getInput(1).text();
		}

		BitwiseShiftLeft* BitwiseShiftLeft::clone() const
		{
			return new BitwiseShiftLeft();
		}
		std::string BitwiseShiftLeft::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " << " + getInput(1).text();
		}

		BitwiseShiftRight* BitwiseShiftRight::clone() const
		{
			return new BitwiseShiftRight();
		}
		std::string BitwiseShiftRight::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " >> " + getInput(1).text();
		}

	} /* namespace nodes */
} /* namespace avocado */

