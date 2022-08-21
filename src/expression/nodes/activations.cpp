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

		Sigmoid* Sigmoid::clone() const
		{
			return new Sigmoid();
		}
		std::string Sigmoid::toString() const
		{
			return this->text() + " = sigmoid(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> Sigmoid::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto y = e.view(this);
			auto dx = dy * y * (e.one() - y);
			return std::vector<node_reference>( { dx });
		}

		ReLU* ReLU::clone() const
		{
			return new ReLU();
		}
		std::string ReLU::toString() const
		{
			return this->text() + " = relu(" + getInput(0).text() + ")";
		}
		std::vector<node_reference> ReLU::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto y = e.view(this);
			auto dx = e.select(y > e.zero(), dy, e.zero());
			return std::vector<node_reference>( { dx });
		}

	} /* namespace nodes */
} /* namespace avocado */

