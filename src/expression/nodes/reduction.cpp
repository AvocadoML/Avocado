/*
 * reduction.cpp
 *
 *  Created on: Aug 2, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/nodes/reduction.hpp>
#include <Avocado/expression/Expression.hpp>
#include <Avocado/core/error_handling.hpp>

namespace
{
	std::string axes_to_string(const std::vector<int> &axes)
	{
		std::string result;
		for (size_t i = 0; i < axes.size(); i++)
		{
			if (i != 0)
				result += ", ";
			result += std::to_string(axes[i]);
		}
		return result;
	}
}

namespace avocado
{
	namespace nodes
	{

		ReduceAdd::ReduceAdd(std::vector<int> axes) :
				Reduction(axes)
		{
		}
		ReduceAdd* ReduceAdd::clone() const
		{
			return new ReduceAdd(m_axes);
		}
		std::string ReduceAdd::toString() const
		{
			return this->text() + " = reduce_add(" + getInput(0).text() + ") over axes: " + axes_to_string(m_axes);
		}
		std::vector<node_reference> ReduceAdd::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
			auto dy = Node::add_gradients(gradients);
			auto dx = dy;
			return std::vector<node_reference>( { dx });
		}

		ReduceMul::ReduceMul(std::vector<int> axes) :
				Reduction(axes)
		{
		}
		ReduceMul* ReduceMul::clone() const
		{
			return new ReduceMul(m_axes);
		}
		std::string ReduceMul::toString() const
		{
			return this->text() + " = reduce_mul(" + getInput(0).text() + ") over axes " + axes_to_string(m_axes);
		}
		std::vector<node_reference> ReduceMul::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
//			auto dy = Node::add_gradients(gradients);

			return std::vector<node_reference>( { });
		}

		ReduceMin::ReduceMin(std::vector<int> axes) :
				Reduction(axes)
		{
		}
		ReduceMin* ReduceMin::clone() const
		{
			return new ReduceMin(m_axes);
		}
		std::string ReduceMin::toString() const
		{
			return this->text() + " = reduce_min(" + getInput(0).text() + ") over axes " + axes_to_string(m_axes);
		}
		std::vector<node_reference> ReduceMin::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
//			auto dy = Node::add_gradients(gradients);

			return std::vector<node_reference>( { });
		}

		ReduceMax::ReduceMax(std::vector<int> axes) :
				Reduction(axes)
		{
		}
		ReduceMax* ReduceMax::clone() const
		{
			return new ReduceMax(m_axes);
		}
		std::string ReduceMax::toString() const
		{
			return this->text() + " = reduce_max(" + getInput(0).text() + ") over axes " + axes_to_string(m_axes);
		}
		std::vector<node_reference> ReduceMax::getBackprop(Expression &e, const std::vector<node_reference> &gradients) const
		{
//			auto dy = Node::add_gradients(gradients);

			return std::vector<node_reference>( { });
		}

		ReduceAnd::ReduceAnd(std::vector<int> axes) :
				Reduction(axes)
		{
		}
		ReduceAnd* ReduceAnd::clone() const
		{
			return new ReduceAnd(m_axes);
		}
		std::string ReduceAnd::toString() const
		{
			return this->text() + " = reduce_and(" + getInput(0).text() + ") over axes " + axes_to_string(m_axes);
		}

		ReduceOr::ReduceOr(std::vector<int> axes) :
				Reduction(axes)
		{
		}
		ReduceOr* ReduceOr::clone() const
		{
			return new ReduceOr(m_axes);
		}
		std::string ReduceOr::toString() const
		{
			return this->text() + " = reduce_or(" + getInput(0).text() + ") over axes " + axes_to_string(m_axes);
		}

	} /* namespace nodes */
} /* namespace avocado */

