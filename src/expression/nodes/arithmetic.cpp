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

		Negation* Negation::clone() const
		{
			return new Negation();
		}
		std::string Negation::toString() const
		{
			return this->text() + " = -" + getInput(0).text();
		}
		std::vector<node_reference> Negation::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto dx = -dy;
			return std::vector<node_reference>( { dx });
		}

		Addition* Addition::clone() const
		{
			return new Addition();
		}
		std::string Addition::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " + " + getInput(1).text();
		}
		std::vector<node_reference> Addition::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto dx1 = dy;
			auto dx2 = dy;
			return std::vector<node_reference>( { dx1, dx2 });
		}

		Subtraction* Subtraction::clone() const
		{
			return new Subtraction();
		}
		std::string Subtraction::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " - " + getInput(1).text();
		}
		std::vector<node_reference> Subtraction::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto dx1 = dy;
			auto dx2 = -dy;
			return std::vector<node_reference>( { dx1, dx2 });
		}

		Multiplication* Multiplication::clone() const
		{
			return new Multiplication();
		}
		std::string Multiplication::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " * " + getInput(1).text();
		}
		std::vector<node_reference> Multiplication::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x1 = e.view(m_inputs.at(0));
			auto x2 = e.view(m_inputs.at(1));
			auto dx1 = dy * x2;
			auto dx2 = dy * x1;
			return std::vector<node_reference>( { dx1, dx2 });
		}

		Division* Division::clone() const
		{
			return new Division();
		}
		std::string Division::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " / " + getInput(1).text();
		}
		std::vector<node_reference> Division::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x1 = e.view(m_inputs.at(0));
			auto x2 = e.view(m_inputs.at(1));
			auto dx1 = dy / x2;
			auto dx2 = -dy * x1 / e.square(x2);
			return std::vector<node_reference>( { dx1, dx2 });
		}

		Modulo* Modulo::clone() const
		{
			return new Modulo();
		}
		std::string Modulo::toString() const
		{
			return this->text() + " = " + getInput(0).text() + " % " + getInput(1).text();
		}
		std::vector<node_reference> Modulo::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x1 = e.view(m_inputs.at(0));
			auto x2 = e.view(m_inputs.at(1));
			auto dx1 = dy;
			auto dx2 = -dy * e.floor(x1 / x2);
			return std::vector<node_reference>( { dx1, dx2 });
		}

	} /* namespace nodes */
} /* namespace avocado */

