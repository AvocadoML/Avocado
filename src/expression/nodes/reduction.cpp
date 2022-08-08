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
		Expression ReduceAdd::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			result.output(dy);
			return result;
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
		Expression ReduceMul::getBackprop() const
		{
			Expression result;
			//			auto dy = result.input();
			//			auto x1 = result.view(m_inputs.at(0));
			//			auto x2 = result.view(m_inputs.at(1));
			//			result.output(dy / x2);
			//			result.output(-dy * x1 / result.square(x2));
			return result;
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
		Expression ReduceMin::getBackprop() const
		{
			Expression result;
			//			auto dy = result.input();
			//			auto x1 = result.view(m_inputs.at(0));
			//			auto x2 = result.view(m_inputs.at(1));
			//			result.output(dy / x2);
			//			result.output(-dy * x1 / result.square(x2));
			return result;
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
		Expression ReduceMax::getBackprop() const
		{
			Expression result;
			//			auto dy = result.input();
			//			auto x1 = result.view(m_inputs.at(0));
			//			auto x2 = result.view(m_inputs.at(1));
			//			result.output(dy / x2);
			//			result.output(-dy * x1 / result.square(x2));
			return result;
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

