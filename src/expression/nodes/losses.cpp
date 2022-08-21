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
		std::vector<node_reference> MeanSquareLoss::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto x1 = e.view(m_inputs.at(0));
			auto x2 = e.view(m_inputs.at(1));
			auto dx1 = dy * (x1 - x2);
			auto dx2 = dy * (x2 - x1);
			return std::vector<node_reference>( { dx1, dx2 });
		}

	} /* namespace nodes */
} /* namespace avocado */
