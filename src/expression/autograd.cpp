/*
 * autograd.cpp
 *
 *  Created on: Aug 13, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/autograd.hpp>
#include <Avocado/expression/Expression.hpp>
#include <Avocado/expression/nodes/Node.hpp>

#include <cassert>

namespace avocado
{

	void Autograd::join(Expression &prev, Expression &next)
	{
		assert(prev.m_outputs.size() == next.m_inputs.size());


	}
	/*
	 * public
	 */
	Expression Autograd::getBackward(const Expression &e)
	{
		std::vector<Expression> backwards(e.m_list_of_nodes.size());
		for (size_t i = 0; i < e.m_list_of_nodes.size(); i++)
		{
			Expression backprop;
			if (e.m_list_of_nodes[i]->numberOfOutputs() > 1)
			{

			}
//			Expression backprop = e.m_list_of_nodes[i]->getBackprop();
		}
		Expression result;
		return result;
	}

} /* namespace avocado */

