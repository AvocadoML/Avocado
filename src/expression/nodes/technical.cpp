/*
 * technical.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/nodes/technical.hpp>
#include <Avocado/expression/Expression.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/core/error_handling.hpp>

namespace avocado
{
	namespace nodes
	{

		Input::Input(const Shape &shape)
		{
			m_output_shape = shape;
		}
		std::string Input::toString() const
		{
			return this->text() + " <- input";
		}

		Target::Target(const Shape &shape)
		{
			m_output_shape = shape;
		}
		std::string Target::toString() const
		{
			return this->text() + " <- target";
		}

		void Output::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			m_output_shape = getInput(0).getOutputShape();
		}
		std::string Output::toString() const
		{
			return getInput(0).text() + " -> output";
		}

		void Loss::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			m_output_shape = getInput(0).getOutputShape();
		}
		std::string Loss::toString() const
		{
			return getInput(0).text() + " -> loss";
		}
		Expression Loss::getBackprop() const
		{
			Expression result;
			result.output(result.one());
			return result;
		}

		void Metric::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			m_output_shape = getInput(0).getOutputShape();
		}
		std::string Metric::toString() const
		{
			return getInput(0).text() + " -> metric";
		}

		void View::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			m_output_shape = getInput(0).getOutputShape();
		}
		std::string View::toString() const
		{
			return this->text() + " = view of " + getInput(0).text();
		}

		std::string Identity::toString() const
		{
			return this->text() + " = " + getInput(0).text();
		}
		Expression Identity::getBackprop() const
		{
			Expression result;
			result.output(result.input(this->getOutputShape()));
			return result;
		}

		/*
		 * Constant values
		 */
		void Zero::calculateOutputShape()
		{
			m_output_shape = { 1 };
		}
		std::string One::toString() const
		{
			return this->text() + " = 1";
		}

		void One::calculateOutputShape()
		{
			m_output_shape = { 1 };
		}
		std::string Zero::toString() const
		{
			return this->text() + " = 0";
		}

		Constant::Constant(double value) :
				m_values( { value })
		{
		}
		Constant::Constant(const std::vector<double> &values) :
				m_values(values)
		{
		}
		void Constant::calculateOutputShape()
		{
			m_output_shape = { static_cast<int>(m_values.size()) };
		}
		std::string Constant::toString() const
		{
			std::string result = this->text() + " = ";
			if (m_values.size() == 1)
				result += std::to_string(m_values[0]);
			else
			{
				result += "{";
				for (size_t i = 0; i < m_values.size(); i++)
				{
					if (i != 0)
						result += ", ";
					result += std::to_string(m_values[i]);
				}
				result += "}";
			}
			return result;
		}

		/*
		 * Selection
		 */
		void Select::calculateOutputShape()
		{
			if (numberOfInputs() != 3)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly three inputs");
			m_output_shape = getShapeAfterBroadcasting(getInput(1).getOutputShape(), getInput(2).getOutputShape());
		}
		std::string Select::toString() const
		{
			return this->text() + " = select(" + getInput(0).text() + ", " + getInput(1).text() + ", " + getInput(2).text() + ")";
		}
		Expression Select::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x = result.view(m_inputs.at(0));
			result.output(result.zero());
			result.output(result.select(x, dy, result.zero()));
			result.output(result.select(x, result.zero(), dy));
			return result;
		}

	} /* namespace nodes */
} /* namespace avocado */

